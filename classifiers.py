import csv
import logging
from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np
import os

GENRE_WEIGHTS = {
    'classical': 1,
    'country': 1,
    'edm_dance': 1,
    'jazz': 1,
    'kids': 1.1,
    'latin': 1.25,
    'metal': 1,
    'pop': 1.25,
    'rnb': 1.25,
    'rock': 1
}


class ConfusionMatrix:
    def __init__(self):
        pass

    def add_genres(self, actual_genre, predicted_genre):
        pass


class SongClassifier:
    __metaclass__ = ABCMeta

    def test(self, songs, genres):
        """
        Test on a data set of songs and known genres.

        :param songs: the songs to test on
        :param genres: the genres of the songs
        """
        num_matches = 0
        confusion_matrix = ConfusionMatrix()
        for song, actual_genre in zip(songs, genres):
            predicted_genre = self.classify(song)
            logging.info('Predicted genre: {}'.format(predicted_genre))
            confusion_matrix.add_genres(actual_genre, predicted_genre)
            if actual_genre == predicted_genre:
                num_matches += 1
        return num_matches, confusion_matrix

    @abstractmethod
    def classify(self, song):
        """
        Classify a particular song.

        :param song: the song to classify
        """
        raise NotImplementedError

    def predict_directory(self, directory_name, result_file_name):
        """
        Predict the genres of all songs in the given directory, saving this data in a file. Note that the genres of the
        songs are not known beforehand.

        :param directory_name:
        :param result_file_name:
        """
        with open(result_file_name, 'ab') as f:
            writer = csv.writer(f)
            writer.writerow(('id', 'category'))
            for song_id in os.listdir(directory_name):
                song = pd.read_csv('{}{}'.format(directory_name, song_id))
                predicted_genre = self.classify(song)
                logging.info('Predicted genre: {}'.format(predicted_genre))
                writer.writerow((song_id, predicted_genre))


class GaussianSongClassifier(SongClassifier):
    def __init__(self, genre_models):
        self.genre_models = genre_models

    def classify(self, song):
        genre_errors = {}
        for model in self.genre_models:
            genre = model.genre
            error = 0
            for feature_vector in song.values:
                error += model.compute_error(feature_vector)
            genre_errors[genre] = error
        return min(genre_errors, key=genre_errors.get)

    class GaussianGenreModel:
        def __init__(self, genre, song_samples):
            self.genre = genre
            self.mean_vector = np.mean(song_samples, axis=0)
            self.covariance_matrix = np.cov(song_samples, rowvar=False)
            self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)

        def compute_error(self, x):
            a = x - self.mean_vector
            return a.dot(self.inv_covariance_matrix).dot(a)


class KnnClassifier(SongClassifier):
    def __init__(self, k, data):
        self.k = k
        self.data = data

    def classify(self, song):
        genre_frequencies = {}
        for feature_vector in song.values:
            genres = self.data.query(feature_vector, self.k)
            for genre in genres:
                genre_frequencies[genre] = genre_frequencies.get(genre, 0) + 1
        return max(genre_frequencies, key=genre_frequencies.get)


    class KnnDataStructure:
        __metaclass__ = ABCMeta

        @abstractmethod
        def query(self, feature_vector, k):
            raise NotImplementedError
