from __future__ import division

import logging
import time

import pandas as pd

from classifiers import GaussianSongClassifier, KnnSongClassifier, KDTreeDataStructure


def load_labels():
    return pd.read_csv('song_data/labels.csv')


def get_training_songs_genres():
    logging.info('Building training and test set.')
    labels_genres = load_labels()
    song_ids = labels_genres['id'].values
    songs = []
    for song_id in song_ids:
        song = pd.read_csv('song_data/training/{}'.format(song_id)).values
        songs.append(song)
    genres = labels_genres['category'].values
    return songs, genres


def split_in_two(l, first=50, second=50):
    split_index = int((first / (first + second)) * len(l))
    return l[:split_index], l[split_index:]


def classify_songs_gaussian():
    songs, genres = get_training_songs_genres()

    training_songs, test_songs = split_in_two(songs)
    training_genres, test_genres = split_in_two(genres)

    classifier = GaussianSongClassifier(training_songs, training_genres)

    num_matches, confusion_matrix = classifier.test(test_songs, test_genres)
    confusion_matrix.save_to_csv('report/csv/confusion_gaussian.csv')
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


def test_songs_knn(k):
    songs, genres = get_training_songs_genres()

    training_songs, test_songs = split_in_two(songs)
    training_genres, test_genres = split_in_two(genres)

    classifier = KnnSongClassifier(k, training_songs, training_genres)

    num_matches, confusion_matrix = classifier.test(test_songs, test_genres)
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


PREDICTION_DIRECTORY = 'song_data/test/'


def predict_songs_knn(k):
    songs, genres = get_training_songs_genres()

    classifier = GaussianSongClassifier(songs, genres)

    classifier.predict_directory(PREDICTION_DIRECTORY, 'song_data/test_labels_knn.csv')


if __name__ == '__main__':
    t = time.time()

    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO)

    classify_songs_gaussian()
    # test_songs_knn(3)
    # predict_songs_knn(3)
    # TODO: predict songs Gaussian

    logging.info('Total runtime: {}'.format(time.time() - t))
