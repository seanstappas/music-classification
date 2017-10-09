import csv

import pandas as pd

GENRE_TRAINING = "song_data/labels_train.csv"
GENRE_TEST = "song_data/labels_test.csv"
GENRES = ['classical', 'country', 'edm_dance', 'jazz', 'kids', 'latin', 'metal', 'pop', 'rnb', 'rock']


def organize_test_training_data():
    labels = pd.read_csv('song_data/labels.csv')
    with open(GENRE_TRAINING, 'ab') as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'category'))
        for song_id, genre in labels.values:
            print('Genre: {}, song_id: {}'.format(genre, song_id))
            genre_id = GENRES.index(genre)
            song = pd.read_csv('song_data/training/{}'.format(song_id))
            for vector in song.values:
                lst = vector.tolist()
                lst.append(genre_id)
                writer.writerow(lst)


if __name__ == '__main__':
    organize_test_training_data()
