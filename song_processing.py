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
from config import Config
import threading
import time
import random


cfg = Config("config.yaml")


class Song:
    def __init__(self, song_alias, sr=cfg.SR, db_name="song_detection.db"):
        """Loads a song into memory"""

        self.database = DBManagement(db_name)
        url_start = "https://www.youtube.com/watch?v="
        self.youtube_id = self.database.get_song_youtube_id(song_alias)

        if self.youtube_id is None:
            song_data = self.get_song_data(song_alias)
            duration = song_data["duration"]
            if cfg.MAX_SONG_DURATION < duration:
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
        """Gets all fingerprints and stores them to a database"""

        window_size = cfg.WINDOW_SIZE
        step = cfg.WINDOW_SIZE // 10
        hashes = Song.get_fingerprints(self.y, window_size, step)
        timestamps = [i / step for i in range(len(hashes))]

        self.database.store_fingerprint(self.youtube_id, hashes, timestamps)

    @staticmethod
    def get_fingerprints(y, window_size, step):
        spectrogram = Song.get_spectrogram(y, window_size, step)

        hashes = Song.get_hashes0(spectrogram)
        hashes = [str(int(element)) for element in hashes]

        return hashes

    @staticmethod
    def get_spectrogram(y, window_size, step, sr=cfg.SR):
        def sliding_windows(x, window_size, step_size):
            shape = ((x.size - window_size) // step_size + 1, window_size)
            strides = (x.strides[0] * step_size, x.strides[0])

            return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        # gets windows if the input size is longer than window size
        if window_size == y.size and step == y.size:
            windows = [y]
        else:
            windows = sliding_windows(y, window_size, step)

        # applies hann window to avoid leakage
        x = np.arange(0, window_size)
        hann_window = np.power(np.sin(np.pi * x / window_size), 2)

        windows = windows * hann_window

        # applies rfft
        X = np.abs(1 / window_size * np.fft.rfft(windows, axis=1))
        frequencies = np.fft.rfftfreq(window_size, 1 / sr)
        mask = frequencies < cfg.MAX_FREQUENCY
        X = X[:, mask]

        return X

    @staticmethod
    def get_hashes0(
        spectrogram,
        binary_powers=np.array(
            [2**i for i in range(cfg.N_CHANNELS * cfg.N_SUBCHANNELS)], dtype=np.uint64
        ),
    ):
        """Calculates hashes from the first maximum in a subchannel"""

        n_samples, freq_length = spectrogram.shape
        step = freq_length // cfg.N_CHANNELS

        # initializes an array to store locations of maximums for each sample
        maximums = np.zeros((n_samples, cfg.N_CHANNELS * cfg.N_SUBCHANNELS))

        for start in range(0, freq_length, step):
            end = start + step
            channel = spectrogram[:, start:end]

            # calculates the subchannel index 
            subchannel_index = (
                np.argmax(channel, axis=1) / step * cfg.N_SUBCHANNELS
            ).astype(np.uint64)
            # recalculates the global index (with all channels)
            fingeprint = int(start / step * cfg.N_SUBCHANNELS) + subchannel_index

            # stores locations for every sample
            for i in range(len(fingeprint)):
                maximums[i, int(fingeprint[i])] = 1

        hashes = np.sum(maximums * binary_powers, axis=1, dtype=np.uint64)

        return hashes

    @staticmethod
    def get_hashes1(
        spectrogram,
        binary_powers=np.array(
            [2**i for i in range(cfg.N_CHANNELS * cfg.N_SUBCHANNELS)], dtype=np.uint64
        ),
    ):
        """
            Calculates hashes from all maximums in a subchannel. 
            The mask condition can also be changed to a desired condition.
            For example an average or 95% of the maximum...
        """

        n_samples, freq_length = spectrogram.shape
        step = freq_length // cfg.N_CHANNELS

        # initializes an array to store locations of maximums for each sample
        maximums = np.zeros((n_samples, cfg.N_CHANNELS * cfg.N_SUBCHANNELS), dtype=np.uint8)

        for channel_ind in range(cfg.N_CHANNELS):
            start = channel_ind * step
            end = start + step
            channel = spectrogram[:, start:end]

            # finds indicies based on the mask condition
            max_vals = np.max(channel, axis=1, keepdims=True)
            mask = max_vals == channel
            Y, X = np.where(mask)

            # calculates the subchannel index 
            subchannel_index = (X / step * cfg.N_SUBCHANNELS).astype(np.uint64)
            # recalculates the global index (with all channels)
            fingerprint = channel_ind * cfg.N_SUBCHANNELS + subchannel_index

            # stores locations for every sample
            for y in range(len(fingerprint)):
                maximums[int(Y[y]), fingerprint[y]] = 1

        hashes = np.sum(maximums * binary_powers, axis=1, dtype=np.uint64)

        return hashes

    @staticmethod
    def get_hashes2(
        spectrogram,
        binary_powers=np.array(
            [2**i for i in range(cfg.N_CHANNELS * cfg.N_SUBCHANNELS)], dtype=np.uint64
        ),
    ):
        """Calculates hashes from the mean value of all subchannel maximums"""
        n_samples, freq_length = spectrogram.shape
        step = freq_length // cfg.N_CHANNELS


        # finds subchannel maximums
        maximum_values = []
        for channel_ind in range(cfg.N_CHANNELS):
            start = channel_ind * step
            end = start + step
            channel = spectrogram[:, start:end]

            max_vals = np.max(channel, axis=1)
            maximum_values.append(max_vals)

        maximum_values = np.array(maximum_values)
        mean_maximum_values = np.mean(maximum_values, axis=0, keepdims=True)

        # increase the threshold for channels whose maximum is above the mean
        thresholds = (maximum_values < mean_maximum_values).astype(
            np.uint64
        ) + maximum_values

        # initializes an array to store locations of maximums for each sample
        maximums = np.zeros((n_samples, cfg.N_CHANNELS * cfg.N_SUBCHANNELS), dtype=np.uint8)

        for channel_ind in range(cfg.N_CHANNELS):
            start = channel_ind * step
            end = start + step
            channel = spectrogram[:, start:end]

            # finds indicies based on the mask condition
            mask = thresholds[channel_ind][:, np.newaxis] <= channel
            Y, X = np.where(mask)

            # calculates the subchannel index 
            subchannel_index = (X / step * cfg.N_SUBCHANNELS).astype(np.uint64)
            # recalculates the global index (with all channels)
            fingerprint = channel_ind * cfg.N_SUBCHANNELS + subchannel_index

            # stores locations for every sample
            for y in range(len(fingerprint)):
                maximums[int(Y[y]), fingerprint[y]] = 1

        hashes = np.sum(maximums * binary_powers, axis=1, dtype=np.uint64)

        return hashes

    @staticmethod
    def get_hashes3(
        spectrogram,
        binary_powers=np.array(
            [2**i for i in range(cfg.N_CHANNELS * cfg.N_SUBCHANNELS)], dtype=np.uint64
        ),
    ):
        """Calculates hashes from the first maximum in a subchannel with multiple samples"""

        n_samples, freq_length = spectrogram.shape
        step = freq_length // cfg.N_CHANNELS

        # initializes an array to store locations of maximums for each sample
        maximums = np.zeros((n_samples - cfg.N_DATAPOINTS + 1, cfg.N_CHANNELS * cfg.N_SUBCHANNELS))

        for sample_start in range(0, n_samples - cfg.N_DATAPOINTS + 1):
            for freq_start in range(0, freq_length, step):
                freq_end = freq_start + step
                channel = spectrogram[
                    sample_start : sample_start + cfg.N_DATAPOINTS, freq_start:freq_end
                ]

                # find maximums on axis 1 (subchannel for every sample seperately)
                maximum_axis1 = (
                    np.argmax(channel, axis=1) / step * cfg.N_SUBCHANNELS
                ).astype(np.uint64)
                # finds maximums on axis 0 (compares subchannel maximums through samples)
                maximum_axis0 = np.argmax(channel[np.arange(channel.shape[0]), maximum_axis1])
                subchannel_index = maximum_axis1[maximum_axis0]

                # recalculates the global index (with all channels)
                fingeprint = int(freq_start / step * cfg.N_SUBCHANNELS) + subchannel_index

                maximums[sample_start, int(fingeprint)] = 1

        hashes = np.sum(maximums * binary_powers, axis=1, dtype=np.uint64)

        return hashes

    @staticmethod
    def generate_hash_iterations(hashed):
        """
            Generates hashes with one incorrect channel.
            An attempt of error correction for hashes version 0
        """
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
    def load_data(filepath, sr=None):
        y, sr = librosa.load(filepath, sr=sr)
        return y, sr

    @staticmethod
    def get_song_data(query):
        """Queries youtube search"""
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
        self.n_windows = 0
        self.matched = dict()
        self.fingerprints = dict()
        self.database = DBManagement("song_detection.db")
        self.scores = dict()
        self.window_size = cfg.WINDOW_SIZE
        self.step = cfg.WINDOW_SIZE // 10
        self.window_index = 0
        self.thread = None

    def add_data(self, y):
        self.data = np.concatenate((self.data, y))
        self.n_windows = len(self.data) // self.step

    def get_current_score(self):
        out = []
        total_matches = sum([val[0] for val in self.scores.values()])
        if total_matches != 0:
            ordered = list(self.scores.keys())
            ordered.sort(reverse=True, key=lambda key: self.scores[key])

            out = [
                (key, self.scores[key][0] / total_matches, self.scores[key])
                for key in ordered[:5]
            ]

        return out

    def start_thread(self):
        """Thread to classify in the background"""
        def worker():
            while self.run_thread:
                while self.n_windows < cfg.N_DATAPOINTS:
                    time.sleep(0.5)
                self.classify()

        self.run_thread = True
        self.thread = threading.Thread(target=worker)
        self.thread.daemon = True
        self.thread.start()

    def stop_thread(self):
        self.run_thread = False
        self.thread.join()

    def classify(self, cut=False):
        def add_time(fingerprint):
            t = self.window_index / cfg.SR
            self.window_index += 1
            if fingerprint not in self.fingerprints.keys():
                self.fingerprints[fingerprint] = []
            self.fingerprints[fingerprint].append(t)

        while cfg.N_DATAPOINTS <= self.n_windows and (self.thread is None or self.run_thread):
            if cfg.N_DATAPOINTS != 1:
                windows = self.data[0:self.step * cfg.N_DATAPOINTS]
                self.data = self.data[self.step:]
                self.n_windows += -1
            else:
                windows = self.data
                self.data = []
                self.n_windows = 0

            fingerprints = Song.get_fingerprints(windows, cfg.WINDOW_SIZE, self.step)

            for fingerprint in fingerprints:
                add_time(fingerprint)
                if fingerprint == "0": continue
                self.find_matches(fingerprint)

            if cut:
                scores = self.get_current_score()
                if len(scores) != 0 and 0.5 < scores[0][1] and 500 < scores[0][2][0]:
                    return

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
        if len(self.matched[youtube_id]) < 100:
            potential_matches = self.matched[youtube_id]
        else:
            potential_matches = random.sample(self.matched[youtube_id], 100)

        for i in range(len(potential_matches)):
            previous = potential_matches[i]
            diff_match = current[3] - previous[3]

            total_diff = 999
            for t1 in self.fingerprints[current[2]]:
                for t0 in self.fingerprints[previous[2]]:
                    diff_og = t1 - t0
                    diff = np.abs(diff_og - diff_match)
                    if diff < total_diff:
                        total_diff = diff

            if total_diff < cfg.MATCH_THRESHOLD:
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
        song = Song(song_name, sr=cfg.SR)
        song.store_fingerprints()
    except Exception:
        print(f"Error processing {i} ({song_name}):", flush=True)
        traceback.print_exc()


def store_songs():
    response = requests.get(cfg.SONGS_URL)
    song_data = response.json()
    song_names = [
        f'{song["songTitle"]} {song["artistTitle"]}' for song in song_data["data"]
    ]

    args = list(enumerate(song_names))

    with multiprocessing.Pool(processes=3) as pool:
        for _ in tqdm(pool.imap(process_song, args), total=len(args)):
            pass
