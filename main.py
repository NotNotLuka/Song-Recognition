from song_processing import Song, store_songs, Classifier
import sys


def find_song(filename):
    y, sr = Song.load_data(filename, sr=22050)

    classifier = Classifier()
    classifier.add_data(y)
    classifier.classify(cut=True)
    scored = classifier.get_current_score()
    print(f"For file {filename}")
    for yotube_id, confidence, _ in scored:
        title = classifier.database.get_song_name(yotube_id)
        print(f"Match found with {title} with confidence {round(confidence * 100, 1)}%")

    return scored

if len(sys.argv) < 2:
    filenames = [
        "tests/example.wav",
        "tests/example1.wav",
        "tests/example2.wav",
        "tests/real_example.wav",
        "tests/Example.wav",
    ]
    for filename in filenames:
        find_song(filename)
elif sys.argv[1] == "store":
    store_songs()
else:
    find_song(sys.argv[1])
