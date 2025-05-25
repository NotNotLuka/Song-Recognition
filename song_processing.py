import librosa
import numpy as np
import matplotlib.pyplot as plt
import yt_dlp
import requests
import multiprocessing
from matplotlib.cm import colors
from pathlib import Path
from sql_queries import DBManagement
import traceback
from tqdm import tqdm


N_CHANNELS = 8
N_SUBCHANNELS = 8
N_datapoints = 5
SR = 22050
MAX_SONG_DURATION = 900
WINDOW_SIZE = SR // 5
MAX_FREQUENCY = 6000
MATCH_THRESHOLD = 0.5


class Song:
    def __init__(self, song_alias, sr=SR, db_name="song_detection.db"):
        self.database = DBManagement(db_name)
        url_start = "https://www.youtube.com/watch?v="
        self.youtube_id = self.database.get_song_youtube_id(song_alias)
        if self.youtube_id is None:
            song_data = self.get_song_data(song_alias)
            duration = song_data["duration"]
            if MAX_SONG_DURATION < duration:
                raise ValueError(f"Found song is too long {duration}s")
            self.youtube_id = song_data["id"]
            if not self.database.check_if_id_exists(self.youtube_id):
                self.database.store_song(
                    song_alias, song_data["title"], song_data["id"]
                )

        filepath = Path(f"media/{self.youtube_id}.mp3")
        if not filepath.exists():
            self.download(url_start + self.youtube_id)

        self.y, self.sr = self.load_data(filepath, sr)

    def store_fingerprints(self):
        window_size = WINDOW_SIZE
        step = WINDOW_SIZE // 10
        hashes = Song.get_fingerprints(self.y, window_size, step)
        timestamps = [i / step for i in range(len(hashes))]

        self.database.store_fingerprint(self.youtube_id, hashes, timestamps)

    @staticmethod
    def get_fingerprints(y, window_size, step):
        spectrogram = Song.get_spectrogram(y, window_size, step)

        hashes = Song.get_hashes3(spectrogram)
        hashes = [str(int(element)) for element in hashes]

        return hashes

    @staticmethod
    def get_spectrogram(y, window_size, step, sr=SR):
        def sliding_windows(x, window_size, step_size):
            shape = ((x.size - window_size) // step_size + 1, window_size)
            strides = (x.strides[0] * step_size, x.strides[0])

            return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        if window_size == y.size and step == y.size:
            windows = [y]
        else:
            windows = sliding_windows(y, window_size, step)

        x = np.arange(0, window_size)
        hann_window = np.power(np.sin(np.pi * x / window_size), 2)

        windows = windows * hann_window

        X = np.abs(1 / window_size * np.fft.rfft(windows, axis=1))
        frequencies = np.fft.rfftfreq(window_size, 1 / sr)
        mask = frequencies < MAX_FREQUENCY
        X = X[:, mask]

        return X

    @staticmethod
    def get_hashes0(
        spectrogram,
        binary_powers=np.array(
            [2**i for i in range(N_CHANNELS * N_SUBCHANNELS)], dtype=np.uint64
        ),
    ):

        n_samples, freq_length = spectrogram.shape
        step = freq_length // N_CHANNELS

        maximums = np.zeros((n_samples, N_CHANNELS * N_SUBCHANNELS))
        for start in range(0, freq_length, step):
            end = start + step
            channel = spectrogram[:, start:end]

            subchannel_index = (
                np.argmax(channel, axis=1) / step * N_SUBCHANNELS
            ).astype(np.uint64)
            fingeprint = int(start / step * N_SUBCHANNELS) + subchannel_index

            for i in range(len(fingeprint)):
                maximums[i, int(fingeprint[i])] = 1

        hashes = np.sum(maximums * binary_powers, axis=1, dtype=np.uint64)

        return hashes

    @staticmethod
    def get_hashes1(
        spectrogram,
        binary_powers=np.array(
            [2**i for i in range(N_CHANNELS * N_SUBCHANNELS)], dtype=np.uint64
        ),
    ):
        n_samples, freq_length = spectrogram.shape
        step = freq_length // N_CHANNELS

        maximums = np.zeros((n_samples, N_CHANNELS * N_SUBCHANNELS), dtype=np.uint8)

        for channel_ind in range(N_CHANNELS):
            start = channel_ind * step
            end = start + step
            channel = spectrogram[:, start:end]

            max_vals = np.max(channel, axis=1, keepdims=True)
            mask = max_vals == channel
            Y, X = np.where(mask)

            subchannel_index = (X / step * N_SUBCHANNELS).astype(np.uint64)
            fingerprint = channel_ind * N_SUBCHANNELS + subchannel_index

            for y in range(len(fingerprint)):
                maximums[int(Y[y]), fingerprint[y]] = 1

        hashes = np.sum(maximums * binary_powers, axis=1, dtype=np.uint64)
        return hashes

    @staticmethod
    def get_hashes2(
        spectrogram,
        binary_powers=np.array(
            [2**i for i in range(N_CHANNELS * N_SUBCHANNELS)], dtype=np.uint64
        ),
    ):
        n_samples, freq_length = spectrogram.shape
        step = freq_length // N_CHANNELS

        maximums = np.zeros((n_samples, N_CHANNELS * N_SUBCHANNELS), dtype=np.uint8)

        maximum_values = []

        for channel_ind in range(N_CHANNELS):
            start = channel_ind * step
            end = start + step
            channel = spectrogram[:, start:end]

            max_vals = np.max(channel, axis=1)
            maximum_values.append(max_vals)

        maximum_values = np.array(maximum_values)
        mean_maximum_values = np.mean(maximum_values, axis=0, keepdims=True)

        thresholds = (maximum_values < mean_maximum_values).astype(
            np.uint64
        ) + maximum_values

        for channel_ind in range(N_CHANNELS):
            start = channel_ind * step
            end = start + step
            channel = spectrogram[:, start:end]
            mask = thresholds[channel_ind][:, np.newaxis] <= channel
            Y, X = np.where(mask)

            subchannel_index = (X / step * N_SUBCHANNELS).astype(np.uint64)
            fingerprint = channel_ind * N_SUBCHANNELS + subchannel_index

            for y in range(len(fingerprint)):
                maximums[int(Y[y]), fingerprint[y]] = 1

        hashes = np.sum(maximums * binary_powers, axis=1, dtype=np.uint64)

        return hashes

    @staticmethod
    def get_hashes3(
        spectrogram,
        binary_powers=np.array(
            [2**i for i in range(N_CHANNELS * N_SUBCHANNELS)], dtype=np.uint64
        ),
    ):

        n_samples, freq_length = spectrogram.shape
        step = freq_length // N_CHANNELS


        maximums = np.zeros((n_samples - N_datapoints + 1, N_CHANNELS * N_SUBCHANNELS))
        for sample_start in range(0, n_samples - N_datapoints + 1):
            for freq_start in range(0, freq_length, step):
                freq_end = freq_start + step
                channel = spectrogram[
                    sample_start : sample_start + N_datapoints, freq_start:freq_end
                ]

                maximum_axis1 = (
                    np.argmax(channel, axis=1) / step * N_SUBCHANNELS
                ).astype(np.uint64)
                maximum_axis0 = np.argmax(channel[np.arange(channel.shape[0]), maximum_axis1])
                subchannel_index = maximum_axis1[maximum_axis0]

                fingeprint = int(freq_start / step * N_SUBCHANNELS) + subchannel_index

                maximums[sample_start, int(fingeprint)] = 1

        hashes = np.sum(maximums * binary_powers, axis=1, dtype=np.uint64)

        return hashes

    @staticmethod
    def generate_hash_iterations(hashed):
        def replace_8_bits(x, pos, val):
            n_bits = 0b11111111
            mask = n_bits << pos
            x &= ~mask
            x |= (val & n_bits) << pos

            return x

        changed = []
        values = [2**i for i in range(8)]
        for pos in range(0, 64, 8):
            for val in values:
                y = replace_8_bits(int(hashed), pos, val)
                changed.append(y)
        return changed

    @staticmethod
    def load_data(filepath: Path, sr=None):
        y, sr = librosa.load(filepath, sr=sr)
        return y, sr

    @staticmethod
    def get_song_data(query):
        ydl_opts = {
            "quiet": True,
            "skip_download": True,
            "extract_flat": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_result = ydl.extract_info(f"ytsearch1:{query}", download=False)
            if (
                search_result
                and "entries" in search_result
                and len(search_result["entries"]) > 0
            ):
                return search_result["entries"][0]
            else:
                return None

    @staticmethod
    def download(url, title="%(id)s"):
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": f"media/{title}.%(ext)s",
            "quiet": False,
            "noplaylist": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])


class Classifier:
    def __init__(self):
        self.data = []
        self.windows = []
        self.matched = dict()
        self.fingerprints = dict()
        self.database = DBManagement("song_detection.db")
        self.scores = dict()
        self.header = None
        self.header_len = -1

    def add_data(self, y, sr=SR):
        window_size = WINDOW_SIZE
        step = WINDOW_SIZE // 10
        start = 0

        out = []

        for end in range(window_size, len(y), step):
            window = y[start:end]
            start += step
            self.windows.append(window)

            if len(self.windows) < N_datapoints: continue

            self.windows = self.windows[-N_datapoints:]
            fingerprint = Song.get_fingerprints(np.concatenate(self.windows), WINDOW_SIZE, step)[0]
            if fingerprint not in self.fingerprints.keys():
                self.fingerprints[fingerprint] = []
            self.fingerprints[fingerprint].append(start / sr)
            if fingerprint == "0":
                continue
            self.find_matches(fingerprint)

            total_matches = sum([val[0] for val in self.scores.values()])
            if total_matches != 0:
                ordered = list(self.scores.keys())
                ordered.sort(reverse=True, key=lambda key: self.scores[key])

                out = [
                    (key, self.scores[key][0] / total_matches, self.scores[key])
                    for key in ordered[:5]
                ]
                if 0.5 < out[0][1] and 500 < out[0][2][0]:
                    return out

        return out

    def find_matches(self, fingerprint):
        output = self.database.search_fingerprint(fingerprint)
        if 200 < len(output):
            return

        for i in range(len(output)):
            out = output[i]
            youtube_id = out[1]

            if youtube_id not in self.matched:
                self.matched[youtube_id] = []
            elif self.matched[youtube_id][-1] == out:
                continue
            else:
                self.evaluate_match(out, youtube_id)
            self.matched[youtube_id].append(out)

    def evaluate_match(self, fingerprint, youtube_id):
        if len(self.matched[youtube_id]) < 2:
            return
        if youtube_id not in self.scores:
            self.scores[youtube_id] = [0, 0]

        current = fingerprint
        for i in range(len(self.matched[youtube_id])):
            previous = self.matched[youtube_id][i]
            diff_match = current[3] - previous[3]

            total_diff = 999
            for t1 in self.fingerprints[current[2]]:
                for t0 in self.fingerprints[previous[2]]:
                    diff_og = t1 - t0
                    diff = np.abs(diff_og - diff_match)
                    if diff < total_diff:
                        total_diff = diff

            if total_diff < MATCH_THRESHOLD:
                self.scores[youtube_id][0] += 1
                self.scores[youtube_id][1] += total_diff


def draw(X, sr, step):
    norm = colors.LogNorm(vmin=X.min() + 1e-15, vmax=X.max())
    plt.imshow(
        X.T,
        aspect="auto",
        origin="lower",
        norm=norm,
        cmap="magma",
        extent=[0, len(X) * step / sr, 0, sr / 2],
    )
    plt.xlabel("t[s]")
    plt.ylabel("f[Hz]")
    plt.show()


def process_song(args):
    i, song_name = args
    try:
        song = Song(song_name, sr=SR)
        song.store_fingerprints()
    except Exception:
        print(f"Error processing {i} ({song_name}):", flush=True)
        traceback.print_exc()


def store_songs():
    response = requests.get(
        "https://gist.githubusercontent.com/keune/0de5c7fb669f7b682874/raw/4aabd7282ee6b58ff886af50489cbcc6c4bd1faf/RollingStone%20Top%20500%20Song"
    )
    song_data = response.json()
    song_names = [
        f'{song["songTitle"]} {song["artistTitle"]}' for song in song_data["data"]
    ]

    args = list(enumerate(song_names))

    with multiprocessing.Pool(processes=8) as pool:
        for _ in tqdm(pool.imap(process_song, args), total=len(args)):
            pass
