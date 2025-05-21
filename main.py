from song_processing import Song, store_songs, Classifier
import sys


def search():
    songs = [
        ("out", "0J2QdDbelmY"),
        ("example", "1lWJXDG2i0A"),
        ("example1", "_7xMfIp-irg"),
        ("example2", "0oox9bJaGJ8"),
        ("real_example", "0iLcqaIS-48"),
        ("Example", "0iLcqaIS-48"),
    ]

    for filename, song in songs[:]:
        y, sr = Song.load_data(f"tests/{filename}.wav", sr=22050)

        classifier = Classifier()
        scored = classifier.add_data(y)

        print(song, [score for score in scored if score[0] == song])


if int(sys.argv[1]):
    search()
else:
    store_songs()
