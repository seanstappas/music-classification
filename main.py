from __future__ import division

import logging
import time

from classifiers import GaussianSongClassifier, KnnSongClassifier
from data_extractor import get_training_songs_genres, PREDICTION_DIRECTORY, DATA_DIRECTORY


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
    confusion_matrix.save_to_csv('report/csv/confusion_knn_{}.csv'.format(k))
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


def predict_songs_gaussian():
    songs, genres = get_training_songs_genres()

    classifier = GaussianSongClassifier(songs, genres)

    classifier.predict_directory(PREDICTION_DIRECTORY, '{}test_labels_gaussian.csv'.format(DATA_DIRECTORY))


def predict_songs_knn(k):
    songs, genres = get_training_songs_genres()

    classifier = KnnSongClassifier(k, songs, genres)

    classifier.predict_directory(PREDICTION_DIRECTORY, '{}test_labels_knn_{}.csv'.format(DATA_DIRECTORY, k))


if __name__ == '__main__':
    t = time.time()

    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO)

    # classify_songs_gaussian()
    test_songs_knn(1)
    # predict_songs_knn(2)
    # TODO: predict songs Gaussian
    # TODO: use k-fold cross-validation

    logging.info('Total runtime: {} s'.format(time.time() - t))
