from __future__ import division

import random
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
from numpy.linalg import inv

from lshash import LSHash

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
        a = x - self.mean_vector
        return a.dot(inv(self.covariance_matrix)).dot(a)


def load_labels():
    return pd.read_csv('song_data/labels.csv')


def classify_gaussian():
    labels = load_labels()
    genre_models = []
    genres = []
    for genre, song_genres_ids in labels.groupby('category'):
        genres.append(genre)
        song_samples = []
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2)):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            song_sample = song.values
            song_samples.append(song_sample)
            # song_samples.append(np.mean(song_sample, axis=0))
        song_samples_matrix = np.vstack(song_samples)  # TODO: Average by length of each song? Now: favors longer songs.
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
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2), num_values):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            actual_genre = classifier.classify(song)
            # actual_genre = random.choice(genres)
            print('Actual genre: {}'.format(actual_genre))
            total_count += 1
            if genre == actual_genre:
                match_count += 1

    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))
    # Matched 429 out of 1511 songs: 28.3917935142%
    # Song average: Matched 182 out of 1511 songs: 12.0450033091%
    # Random: Matched 167 out of 1511 songs: 11.0522832561%
    # Half dataset: Matched 221 out of 758 songs: 29.1556728232%


def classify_nearest_neighbor(k):
    labels = load_labels()
    total_count = 0
    match_count = 0
    for genre, song_genres_ids in labels.groupby('category'):
        print('Expected genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2), num_values):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            actual_genre = classify_neighbors_song(k, song, labels)
            # actual_genre = random.choice(genres)
            print('Actual genre: {}'.format(actual_genre))
            total_count += 1
            if genre == actual_genre:
                match_count += 1

    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))
    # Matched 429 out of 1511 songs: 28.3917935142%
    # Song average: Matched 182 out of 1511 songs: 12.0450033091%
    # Random: Matched 167 out of 1511 songs: 11.0522832561%
    # Half dataset: Matched 221 out of 758 songs: 29.1556728232%


def classify_neighbors_song(k, song, song_labels):
    genre_freqs = {}
    for value in song.values:
        genre = classify_neighbors_vector(k, value, song_labels)
        print('Classified {} for vector.'.format(genre))
        genre_freqs[genre] = genre_freqs.get(genre, 0) + 1
    return max(genre_freqs, key=genre_freqs.get)


song_cache = {}


def classify_neighbors_vector(k, value, song_labels):
    distances_genres = []
    for genre, song_genres_ids in song_labels.groupby('category'):
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2)):
            val = song_genres_ids.values[i]
            song_id = val[0]
            if song_id in song_cache:
                # print('Cache hit')
                song = song_cache[song_id]
            else:
                # print('Cache miss')
                song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
                song_cache[song_id] = song
            for feature_vector in song.values:
                distance = euclidean_distance(value, feature_vector)
                distances_genres.append((distance, genre))
    distances_genres.sort()
    print('Distances: {}'.format(distances_genres))
    genre_freqs = {}
    for i in range(k):
        genre = distances_genres[i][1]
        genre_freqs[genre] = genre_freqs.get(genre, 0) + 1
    return max(genre_freqs, key=genre_freqs.get)


def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


def classify_nearest_neighbor_lsh(k):
    lsh = LSHash(3, 12)
    labels = load_labels()

    for genre, song_genres_ids in labels.groupby('category'):
        print('Indexing genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2)):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            for val in song.values:
                lsh.index(val)

    total_count = 0
    match_count = 0
    for genre, song_genres_ids in labels.groupby('category'):
        print('Expected genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2), num_values):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            for val in song.values:
                print('Cache hit:')
                print(lsh.query(val)[0])
            actual_genre = 'test'
            # actual_genre = classify_neighbors_song_lsh(k, song, labels)
            # actual_genre = random.choice(genres)
            print('Actual genre: {}'.format(actual_genre))
            total_count += 1
            if genre == actual_genre:
                match_count += 1

    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))
    # Matched 429 out of 1511 songs: 28.3917935142%
    # Song average: Matched 182 out of 1511 songs: 12.0450033091%
    # Random: Matched 167 out of 1511 songs: 11.0522832561%
    # Half dataset: Matched 221 out of 758 songs: 29.1556728232%


def classify_neighbors_song_lsh(k, song, song_labels):
    genre_freqs = {}
    for value in song.values:
        genre = classify_neighbors_vector_lsh(k, value, song_labels)
        print('Classified {} for vector.'.format(genre))
        genre_freqs[genre] = genre_freqs.get(genre, 0) + 1
    return max(genre_freqs, key=genre_freqs.get)


def classify_neighbors_vector_lsh(k, value, song_labels):
    distances_genres = []
    for genre, song_genres_ids in song_labels.groupby('category'):
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2)):
            val = song_genres_ids.values[i]
            song_id = val[0]
            if song_id in song_cache:
                song = song_cache[song_id]
            else:
                song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
                song_cache[song_id] = song
            for feature_vector in song.values:
                distance = euclidean_distance(value, feature_vector)
                distances_genres.append((distance, genre))
    distances_genres.sort()
    print('Distances: {}'.format(distances_genres))
    genre_freqs = {}
    for i in range(k):
        genre = distances_genres[i][1]
        genre_freqs[genre] = genre_freqs.get(genre, 0) + 1
    return max(genre_freqs, key=genre_freqs.get)


class LocalitySensitiveHash:
    def __init__(self, l):
        self.hash_tables = []
        for i in range(l):
            self.hash_tables.append({})

    def classify_vector(self, vector):
        pass







if __name__ == '__main__':
    # classify_gaussian()
    # classify_nearest_neighbor(5)  # TODO: Implement LSH or k-d tree (too slow now...)
    classify_nearest_neighbor_lsh(5)
