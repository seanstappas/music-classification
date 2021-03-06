import csv
import logging
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import sklearn.neighbors as nb
from sklearn import svm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


class SongClassifier:
    """
    Classifies songs by their genre.
    """
    __metaclass__ = ABCMeta

    def test(self, songs, genres):
        """
        Test on a data set of songs and known genres.

        :param songs: the songs to test on
        :param genres: the genres of the songs
        """
        logging.info('Starting testing.')
        num_matches = 0
        confusion_matrix = ConfusionMatrix(genres)
        for song, actual_genre in zip(songs, genres):
            predicted_genre = self.classify(song)
            logging.info('Actual genre: {}, predicted genre: {}'.format(actual_genre, predicted_genre))
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
        logging.info('Starting prediction.')
        with open(result_file_name, 'ab') as f:
            writer = csv.writer(f)
            writer.writerow(('id', 'category'))
            for song_id in os.listdir(directory_name):
                song = pd.read_csv('{}{}'.format(directory_name, song_id)).values
                predicted_genre = self.classify(song)
                logging.info('Predicted genre: {}'.format(predicted_genre))
                writer.writerow((song_id, predicted_genre))


class GaussianSongClassifier(SongClassifier):
    """
    Classifies songs by modeling each genre as a Gaussian distribution.
    """
    def __init__(self, songs, genres):
        logging.info('Constructing Gaussian classifier.')
        genres_to_songs = {}
        for genre in genres:
            genres_to_songs[genre] = []
        for song, genre in zip(songs, genres):
            genres_to_songs[genre].append(song)

        self.genre_models = []
        for genre, songs in genres_to_songs.iteritems():
            self.genre_models.append(GaussianGenreModel(songs, genre))

    def classify(self, song):
        genre_errors = {}
        for model in self.genre_models:
            genre = model.genre
            error = 0
            for feature_vector in song:
                error += model.compute_error(feature_vector)
            genre_errors[genre] = error
        return min(genre_errors, key=genre_errors.get)

    def classify_average(self, song):
        genre_errors = {}
        for model in self.genre_models:
            genre = model.genre
            average_vector = np.mean(song)
            error = model.compute_error(average_vector)
            genre_errors[genre] = error
        return min(genre_errors, key=genre_errors.get)


class GaussianGenreModel:
    """
    Gaussian model for a genre, represented by the mean and covariance matrices of the song feature vectors.
    """
    def __init__(self, songs, genre):
        logging.info('Constructing Gaussian model for genre {}.'.format(genre))
        stacked_songs = np.vstack(songs)
        self.mean_vector = np.mean(stacked_songs, axis=0)
        self.covariance_matrix = np.cov(stacked_songs, rowvar=False)
        self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)
        self.genre = genre

    def compute_error(self, x):
        a = x - self.mean_vector
        return a.dot(self.inv_covariance_matrix).dot(a)


class KnnSongClassifier(SongClassifier):
    """
    Classifies songs by modeling each genre as a Gaussian distribution.
    """
    def __init__(self, k, songs, genres, data_structure='kd_tree'):
        logging.info('Constructing kNN classifier (k={}).'.format(k))
        self.k = k
        if data_structure == 'kd_tree':
            self.data = KDTreeDataStructure(songs, genres)
        elif data_structure == 'simple':
            self.data = SimpleDataStructure(songs, genres)
        elif data_structure == 'average':
            self.data = AverageDataStructure(songs, genres)
        else:
            raise ValueError('Invalid knn data structure.')

    def classify(self, song):
        genre_frequencies = {}
        for feature_vector in song:
            genres = self.data.query(feature_vector, self.k)
            for genre in genres:
                genre_frequencies[genre] = genre_frequencies.get(genre, 0) + 1
        return max(genre_frequencies, key=genre_frequencies.get)


class KnnDataStructure:
    """
    Data structure to query nearest neighbours for the kNN classifier.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def query(self, feature_vector, k):
        raise NotImplementedError


class SimpleDataStructure(KnnDataStructure):
    """
    Simple data structure which queries all the feature vectors to find the k nearest ones.
    """
    def __init__(self, songs, genres):
        self.feature_vectors = np.vstack(songs)
        self.genres = []
        for song, genre in zip(songs, genres):
            for _ in song:
                self.genres.append(genre)

    def query(self, feature_vector, k):
        genre_distances = np.empty([len(self.feature_vectors), 1])
        indx = 0
        for v, genre in zip(self.feature_vectors, self.genres):
            genre_distances[indx] = euclidean_distance(v, feature_vector)
            indx += 1
        logging.debug('genre_distances: {}'.format(genre_distances))
        logging.debug('genre_distances size: {}'.format(genre_distances.shape))
        indices = np.argpartition(genre_distances, k, axis=None)[:k]
        logging.debug('Indices: {}'.format(indices))
        return [self.genres[i] for i in indices]


class AverageDataStructure(KnnDataStructure):
    """
    Faster data structure which queries all the average feature vectors to find the k nearest ones.
    """
    def __init__(self, songs, genres):
        self.feature_vectors = []
        self.genres = []
        for song, genre in zip(songs, genres):
            self.genres.append(genre)
            self.feature_vectors.append(np.mean(song))

    def query(self, feature_vector, k):
        genre_distances = np.empty([len(self.feature_vectors), 1])
        indx = 0
        for v, genre in zip(self.feature_vectors, self.genres):
            genre_distances[indx] = euclidean_distance(v, feature_vector)
            indx += 1
        logging.debug('genre_distances: {}'.format(genre_distances))
        logging.debug('genre_distances size: {}'.format(genre_distances.shape))
        indices = np.argpartition(genre_distances, k, axis=None)[:k]
        logging.debug('Indices: {}'.format(indices))
        return [self.genres[i] for i in indices]


def euclidean_distance(a, b):
    """
    Compute the euclidean distance between two numpy vectors

    :param a: the first vector
    :param b: the second vector
    :return: the distance between the vec
    """
    return np.linalg.norm(a - b)


class KDTreeDataStructure(KnnDataStructure):
    """
    K-d tree structure to search for nearest neighbours.
    """
    def __init__(self, songs, genres):
        self.kd_tree = nb.KDTree(np.vstack(songs))
        self.genres = []
        for song, genre in zip(songs, genres):
            for _ in song:
                self.genres.append(genre)

    def query(self, feature_vector, k):
        indices = self.kd_tree.query([feature_vector], k, return_distance=False)
        logging.debug('Neighbours: {}'.format(indices))
        return [self.genres[i] for i in indices[0]]


class ConfusionMatrix:
    """
    Represents a confusion matrix, specifying the predicted genres for each actual genre.
    """
    def __init__(self, genres):
        self.dic = {}
        for g1 in genres:
            self.dic[g1] = {}
            for g2 in genres:
                self.dic[g1][g2] = 0

    def add_genres(self, actual_genre, predicted_genre):
        self.dic[actual_genre][predicted_genre] += 1

    def save_to_csv(self, filename):
        with open(filename, "wb") as f:
            writer = csv.writer(f)
            genres = self.dic.keys()
            writer.writerow([' '] + genres)  # Header
            for genre in genres:
                logging.debug('Predicted genres: {}'.format(self.dic[genre].keys()))
                writer.writerow([genre] + self.dic[genre].values())


class SvmSongClassifier(SongClassifier):
    def __init__(self, songs, genres):
        logging.info('Constructing SVM classifier.')
        self.classifier = svm.SVC(gamma=2, C=1)
        x = np.vstack(songs)
        y = []
        for song, genre in zip(songs, genres):
            for _ in song:
                y.append(genre)
        logging.info('Fitting feature vectors to genres.')
        self.classifier.fit(x, y)

    def classify(self, song):
        genre_frequencies = {}
        for feature_vector in song:
            genre = self.classifier.predict([feature_vector])[0]
            genre_frequencies[genre] = genre_frequencies.get(genre, 0) + 1
        return max(genre_frequencies, key=genre_frequencies.get)


class NaiveBayesSongClassifier(SongClassifier):
    def __init__(self, songs, genres):
        logging.info('Constructing Naive Bayes classifier.')
        self.classifier = GaussianNB()
        x = np.vstack(songs)
        y = []
        for song, genre in zip(songs, genres):
            for _ in song:
                y.append(genre)
        logging.info('Fitting feature vectors to genres.')
        self.classifier.fit(x, y)

    def classify(self, song):
        genre_frequencies = {}
        for feature_vector in song:
            genre = self.classifier.predict([feature_vector])[0]
            genre_frequencies[genre] = genre_frequencies.get(genre, 0) + 1
        return max(genre_frequencies, key=genre_frequencies.get)


class NeuralNetworkSongClassifier(SongClassifier):
    def __init__(self, songs, genres):
        logging.info('Constructing Neural Network classifier.')
        self.classifier = MLPClassifier(hidden_layer_sizes=(12, 30, 12))
        x = np.vstack(songs)
        y = []
        for song, genre in zip(songs, genres):
            for _ in song:
                y.append(genre)
        logging.info('Fitting feature vectors to genres.')
        self.classifier.fit(x, y)

    def classify(self, song):
        genre_frequencies = {}
        for feature_vector in song:
            genre = self.classifier.predict([feature_vector])[0]
            genre_frequencies[genre] = genre_frequencies.get(genre, 0) + 1
        return max(genre_frequencies, key=genre_frequencies.get)


class GaussianProcessSongClassifier(SongClassifier):
    def __init__(self, songs, genres):
        logging.info('Constructing Gaussian Process classifier.')
        self.classifier = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)
        x = np.vstack(songs)
        y = []
        for song, genre in zip(songs, genres):
            for _ in song:
                y.append(genre)
        logging.info('Fitting feature vectors to genres.')
        self.classifier.fit(x, y)

    def classify(self, song):
        genre_frequencies = {}
        for feature_vector in song:
            genre = self.classifier.predict([feature_vector])[0]
            genre_frequencies[genre] = genre_frequencies.get(genre, 0) + 1
        return max(genre_frequencies, key=genre_frequencies.get)


class AdaSongClassifier(SongClassifier):
    def __init__(self, songs, genres):
        logging.info('Constructing ADA classifier.')
        self.classifier = AdaBoostClassifier()
        x = np.vstack(songs)
        y = []
        for song, genre in zip(songs, genres):
            for _ in song:
                y.append(genre)
        logging.info('Fitting feature vectors to genres.')
        self.classifier.fit(x, y)

    def classify(self, song):
        genre_frequencies = {}
        for feature_vector in song:
            genre = self.classifier.predict([feature_vector])[0]
            genre_frequencies[genre] = genre_frequencies.get(genre, 0) + 1
        return max(genre_frequencies, key=genre_frequencies.get)


class QdaSongClassifier(SongClassifier):
    def __init__(self, songs, genres):
        logging.info('Constructing QDA classifier.')
        self.classifier = QuadraticDiscriminantAnalysis()
        x = np.vstack(songs)
        y = []
        for song, genre in zip(songs, genres):
            for _ in song:
                y.append(genre)
        logging.info('Fitting feature vectors to genres.')
        self.classifier.fit(x, y)

    def classify(self, song):
        genre_frequencies = {}
        for feature_vector in song:
            genre = self.classifier.predict([feature_vector])[0]
            genre_frequencies[genre] = genre_frequencies.get(genre, 0) + 1
        return max(genre_frequencies, key=genre_frequencies.get)
