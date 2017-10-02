from abc import ABCMeta, abstractmethod

import pandas as pd
from numpy.linalg import inv


class SongClassifier:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, song):
        raise NotImplementedError

    @abstractmethod
    def classify(self, song):
        raise NotImplementedError


class GaussianSongClassifier(SongClassifier):
    def __init__(self, genre_models):
        self.genre_models = genre_models

    def train(self, song):
        for model in self.genre_models:
            for feature_vector in song:
                model.classify(feature_vector)

    def classify(self, song):
        genre_errors = {}
        for model in self.genre_models:
            genre = model.genre
            for feature_vector in song:
                genre_errors[genre] += model.compute_error(feature_vector)
        return min(genre_errors, key=genre_errors.get)


class GaussianGenreModel:
    def __init__(self, genre, mean_vector, covariance_matrix):
        self.genre = genre
        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix

    def classify(self, x):
        pass

    def compute_error(self, x):
        a = x - self.mean_vector
        return a.dot(inv(self.covariance_matrix)).dot(a.T).item()  # Array of size 1, return its value


def load_labels():
    return pd.read_csv('song_data/labels.csv')


def test_pandas():
    labels = load_labels()
    n = len(labels)

    for row in labels.values:
        for val in row:
            print('Val: {}'.format(val))

    # for i in range(n):
    #     song_id = labels.loc[i, 'id']
    #     category = labels.loc[i, 'category']
    #     song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
    #     print(song)


if __name__ == '__main__':
    test_pandas()
