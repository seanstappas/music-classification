from __future__ import division

from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from numpy.linalg import inv


class SongClassifier:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, song, genre):
        raise NotImplementedError

    @abstractmethod
    def classify(self, song):
        raise NotImplementedError


class GaussianSongClassifier(SongClassifier):
    def __init__(self, genre_models):
        self.genre_models = genre_models

    def train(self, song, genre):
        model = self.genre_models[genre]
        model.update(song)

    def classify(self, song):
        genre_errors = {}
        for model in self.genre_models:
            genre = model.genre
            error = 0
            for feature_vector in song.values:
                error += model.compute_error(feature_vector)
            genre_errors[genre] = error
        return min(genre_errors, key=genre_errors.get)


class NearestNeighbourClassifier(SongClassifier):
    def train(self, song, genre):
        pass

    def classify(self, song):
        pass


class NaiveGaussianGenreModel:
    def __init__(self, genre, song_samples):
        self.genre = genre
        self.mean_vector = np.mean(song_samples, axis=0)
        self.covariance_matrix = np.cov(song_samples, rowvar=False)

    def compute_error(self, x):
        return (x - self.mean_vector).dot(inv(self.covariance_matrix)).dot(x - self.mean_vector)


def load_labels():
    return pd.read_csv('song_data/labels.csv')


def test_pandas():
    labels = load_labels()
    genre_models = []
    for genre, song_genres_ids in labels.groupby('category'):
        song_samples = []
        for val in song_genres_ids.values:
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            song_sample = song.values
            song_samples.append(song_sample)
        song_samples_matrix = np.vstack(song_samples)
        # print('Training genre {}'.format(genre))
        # print('SAMPLES: {}'.format(song_samples_matrix))
        # print('SAMPLES size: {}'.format(song_samples_matrix.shape))
        genre_model = NaiveGaussianGenreModel(genre, song_samples_matrix)
        genre_models.append(genre_model)

    classifier = GaussianSongClassifier(genre_models)

    total_count = 0
    match_count = 0
    for genre, song_genres_ids in labels.groupby('category'):
        print('Expected genre: {}'.format(genre))
        for val in song_genres_ids.values:
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            actual_genre = classifier.classify(song)
            print('Actual genre: {}'.format(actual_genre))
            total_count += 1
            if genre == actual_genre:
                match_count += 1

    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))





    # for row in labels.values:
    #     for val in row:
    #         print('Val: {}'.format(val))

    # for i in range(1):
    #     song_id = labels.loc[i, 'id']
    #     category = labels.loc[i, 'category']
    #     song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
    #     print(song)
    #     # print('Song type: {}'.format(type(song)))
    #     # print('Song values type: {}'.format(type(song.values)))
    #     print('Song {} transpose (row i = values for feature i):'.format(i))
    #     print(song.values.T)
    #     # for feature_vector in song.values:
    #     #     print('feature_vector type: {}'.format(type(feature_vector)))
    #     #     for value in feature_vector:
    #     #         print('value type: {}'.format(type(feature_vector)))


if __name__ == '__main__':
    test_pandas()
