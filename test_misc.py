from __future__ import division

import csv
import logging
import os
import time

import numpy as np
import pandas as pd
import sklearn.neighbors as nb
from lshash import LSHash
from scipy.spatial import KDTree

from classifiers import GaussianSongClassifier, KnnSongClassifier, KDTreeDataStructure, QdaSongClassifier, \
    AdaSongClassifier, GaussianProcessSongClassifier, NeuralNetworkSongClassifier, SvmSongClassifier, \
    NaiveBayesSongClassifier
from data_extractor import DATA_DIRECTORY
from main import split_in_k


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
        song_samples_matrix = np.vstack(song_samples)  # TODO: Average by length of each song? Now: favors longer songs.
        # genre_model = NaiveGaussianGenreModel(genre, song_samples_matrix)
        # genre_models.append(genre_model)

    classifier = GaussianSongClassifier(genre_models)

    total_count = 0
    match_count = 0
    confusion_matrix = {}
    for genre in genres:
        confusion_matrix[genre] = {}
        for genre2 in genres:
            confusion_matrix[genre][genre2] = 0
    for genre, song_genres_ids in labels.groupby('category'):
        print('Expected genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2), num_values):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            predicted_genre = classifier.classify(song)
            print('Predicted genre: {}'.format(predicted_genre))
            confusion_matrix[genre][predicted_genre] += 1
            total_count += 1
            if genre == predicted_genre:
                match_count += 1

    print('Confusion matrix: {}'.format(confusion_matrix))
    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))
    # Matched 429 out of 1511 songs: 28.3917935142%
    # Song average: Matched 182 out of 1511 songs: 12.0450033091%
    # Random: Matched 167 out of 1511 songs: 11.0522832561%
    # Half dataset: Matched 221 out of 758 songs: 29.1556728232%

    # Confusion matrix: {'latin': {'latin': 8, 'classical': 0, 'country': 0, 'rock': 0, 'jazz': 0, 'metal': 0,
    # 'pop': 0, 'edm_dance': 0, 'kids': 0, 'rnb': 67}, 'classical': {'latin': 0, 'classical': 50, 'country': 0,
    # 'rock': 0, 'jazz': 6, 'metal': 0, 'pop': 0, 'edm_dance': 0, 'kids': 14, 'rnb': 0}, 'country': {'latin': 0,
    # 'classical': 0, 'country': 0, 'rock': 0, 'jazz': 0, 'metal': 0, 'pop': 10, 'edm_dance': 0, 'kids': 1,
    # 'rnb': 67}, 'rock': {'latin': 0, 'classical': 0, 'country': 0, 'rock': 0, 'jazz': 0, 'metal': 0, 'pop': 9,
    # 'edm_dance': 0, 'kids': 1, 'rnb': 61}, 'jazz': {'latin': 0, 'classical': 0, 'country': 0, 'rock': 0,
    # 'jazz': 66, 'metal': 0, 'pop': 0, 'edm_dance': 0, 'kids': 3, 'rnb': 13}, 'metal': {'latin': 0, 'classical': 0,
    # 'country': 0, 'rock': 3, 'jazz': 0, 'metal': 0, 'pop': 28, 'edm_dance': 4, 'kids': 10, 'rnb': 29},
    # 'pop': {'latin': 0, 'classical': 0, 'country': 0, 'rock': 0, 'jazz': 0, 'metal': 0, 'pop': 1, 'edm_dance': 0,
    # 'kids': 1, 'rnb': 73}, 'edm_dance': {'latin': 0, 'classical': 0, 'country': 0, 'rock': 0, 'jazz': 0,
    # 'metal': 0, 'pop': 4, 'edm_dance': 0, 'kids': 3, 'rnb': 72}, 'kids': {'latin': 0, 'classical': 3, 'country': 0,
    #  'rock': 0, 'jazz': 12, 'metal': 0, 'pop': 2, 'edm_dance': 0, 'kids': 27, 'rnb': 35}, 'rnb': {'latin': 0,
    # 'classical': 0, 'country': 0, 'rock': 0, 'jazz': 4, 'metal': 0, 'pop': 1, 'edm_dance': 0, 'kids': 1,
    # 'rnb': 69}} Matched 221 out of 758 songs: 29.1556728232%

    # TODO: Create confusion matrix... Why is RnB predicted so often? Maybe take weighted average...


def classify_gaussian_kaggle():
    labels = load_labels()
    genre_models = []
    genres = []
    for genre, song_genres_ids in labels.groupby('category'):
        genres.append(genre)
        song_samples = []
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values)):
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
        # genre_model = NaiveGaussianGenreModel(genre, song_samples_matrix)
        # genre_models.append(genre_model)

    classifier = GaussianSongClassifier(genre_models)

    with open('song_data/test_labels.csv', 'ab') as f:
        writer = csv.writer(f)
        writer.writerow(('id', 'category'))
        for song_id in os.listdir('song_data/test/'):
            song = pd.read_csv('song_data/test/{}'.format(song_id))
            genre = classifier.classify(song)
            print('Predicted genre: {}'.format(genre))
            writer.writerow((song_id, genre))


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
    # Matched x out of y songs: z%


def classify_neighbors_song(k, song, song_labels):
    genre_freqs = {}
    for value in song.values:
        genre = classify_neighbors_vector(k, value, song_labels)
        # print('Classified {} for vector.'.format(genre))
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
    # print('Distances: {}'.format(distances_genres))
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
                lsh.index(val, extra_data=genre)

    total_count = 0
    match_count = 0
    for genre, song_genres_ids in labels.groupby('category'):
        print('Expected genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2), num_values):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            genre_freqs = {}

            split_song = np.array_split(song, 5, axis=0)  # Split song into sections
            for s in split_song:
                avg_song_val = np.mean(s)  # Take average of each section
                neighbours = lsh.query(avg_song_val, num_results=k)
                for neighbour in neighbours:
                    genre = neighbour[0][1]
                    genre_freqs[genre] = genre_freqs.get(genre, 0) + 1

            actual_genre = max(genre_freqs, key=genre_freqs.get)
            print('Predicted genre: {}'.format(actual_genre))
            total_count += 1
            if genre == actual_genre:
                match_count += 1

    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))
    # Matched 378 out of 758 songs: 49.8680738786% (Using average of each song)


class LocalitySensitiveHash:
    # TODO: Create custom LSH
    def __init__(self, l):
        self.hash_tables = []
        for i in range(l):
            self.hash_tables.append({})

    def classify_vector(self, vector):
        pass


class LocalitySensitiveHash:
    # TODO: Create custom LSH
    def __init__(self, l):
        self.hash_tables = []
        for i in range(l):
            self.hash_tables.append({})

    def classify_vector(self, vector):
        pass


class SongSample:
    def __init__(self, data, genre):
        self.data = data
        self.genre = genre

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def __repr__(self):
        return 'Item({}, {}, {})'.format(self.data[0], self.data[1], self.data)


def classify_nearest_neighbor_kd_tree(k):
    labels = load_labels()

    song_samples = []
    indexed_genres = []

    for genre, song_genres_ids in labels.groupby('category'):
        print('Indexing genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2)):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            for val in song.values:
                song_samples.append(val)
                indexed_genres.append(genre)

    kd_tree = KDTree(np.vstack(song_samples))

    total_count = 0
    match_count = 0
    for genre, song_genres_ids in labels.groupby('category'):
        print('Expected genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2), num_values):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            genre_freqs = {}

            split_song = np.array_split(song, 5, axis=0)  # Split song into sections
            for s in split_song:
                avg_song_val = np.mean(s)  # Take average of each section
                _, genre_indices = kd_tree.query(avg_song_val, k)
                for index in genre_indices:
                    genre = indexed_genres[index]
                    genre_freqs[genre] = genre_freqs.get(genre, 0) + 1

            actual_genre = max(genre_freqs, key=genre_freqs.get)
            print('Predicted genre: {}'.format(actual_genre))
            total_count += 1
            if genre == actual_genre:
                match_count += 1

    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))


def classify_nearest_neighbor_kd_tree_sk(k):
    print('k = {}'.format(k))
    labels = load_labels()

    song_samples = []
    indexed_genres = []

    for genre, song_genres_ids in labels.groupby('category'):
        print('Indexing genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2)):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            for val in song.values:
                song_samples.append(val)
                indexed_genres.append(genre)

    kd_tree = nb.KDTree(np.vstack(song_samples))

    total_count = 0
    match_count = 0
    for genre, song_genres_ids in labels.groupby('category'):
        print('Expected genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2), num_values):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            genre_freqs = {}
            # s = np.mean(song)
            # split_song = np.array_split(song, 5, axis=0)  # Split song into sections
            for s in song.values:
                # avg_song_val = np.mean(s)  # Take average of each section
                genre_indices = kd_tree.query([s], k, return_distance=False)
                logging.debug('Length of indexed genres: {}'.format(len(indexed_genres)))
                logging.debug('genre_indices: {}'.format(genre_indices))
                for index in genre_indices[0]:
                    g = indexed_genres[index]
                    genre_freqs[g] = genre_freqs.get(g, 0) + 1

            actual_genre = max(genre_freqs, key=genre_freqs.get)
            print('Predicted genre: {}'.format(actual_genre))
            total_count += 1
            if genre == actual_genre:
                match_count += 1

    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))
    # Average
    # k = 1: Matched 198 out of 758 songs: 26.1213720317%
    # 3: Matched 215 out of 758 songs: 28.364116095%
    # 100: Matched 292 out of 758 songs: 38.5224274406%

    # Non Average:
    # k = 1: Matched 447 out of 758 songs: 58.9709762533%
    # k = 2: Matched 455 out of 758 songs: 60.0263852243%
    # k = 3: Matched 455 out of 758 songs: 60.0263852243%
    # k = 4: Matched 450 out of 758 songs: 59.3667546174%
    # k = 5: Matched 451 out of 758 songs: 59.4986807388%
    # k = 10: Matched 440 out of 758 songs: 58.0474934037%
    # k = 100: Matched 415 out of 758 songs: 54.7493403694%
    # k = 500: Matched 384 out of 758 songs: 50.6596306069%
    # k = 1171: Matched 364 out of 758 songs: 48.0211081794%
    # k = 2000: Matched 346 out of 758 songs: 45.6464379947%


def classify_nearest_neighbor_ball_tree(k):
    labels = load_labels()

    song_samples = []
    indexed_genres = []

    for genre, song_genres_ids in labels.groupby('category'):
        print('Indexing genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2)):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            for val in song.values:
                song_samples.append(val)
                indexed_genres.append(genre)

    ball_tree = nb.BallTree(np.vstack(song_samples))

    total_count = 0
    match_count = 0
    for genre, song_genres_ids in labels.groupby('category'):
        print('Expected genre: {}'.format(genre))
        num_values = len(song_genres_ids.values)
        for i in range(int(num_values / 2), num_values):
            val = song_genres_ids.values[i]
            song_id = val[0]
            song = pd.read_csv('song_data/training/{}'.format(song_id), header=None)
            genre_freqs = {}

            split_song = np.array_split(song, 5, axis=0)  # Split song into sections
            for s in split_song:
                avg_song_val = np.mean(s)  # Take average of each section
                genre_indices = ball_tree.query([avg_song_val], k, return_distance=False)
                for index in genre_indices[0]:
                    genre = indexed_genres[index]
                    genre_freqs[genre] = genre_freqs.get(genre, 0) + 1

            actual_genre = max(genre_freqs, key=genre_freqs.get)
            print('Predicted genre: {}'.format(actual_genre))
            total_count += 1
            if genre == actual_genre:
                match_count += 1

    print('Matched {} out of {} songs: {}%'.format(match_count, total_count, (match_count / total_count) * 100))
    # Average of 5 slices: Matched 297 out of 758 songs: 39.1820580475%


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
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


def test_songs_knn(k):
    songs, genres = get_training_songs_genres()

    training_songs, test_songs = split_in_two(songs)
    training_genres, test_genres = split_in_two(genres)

    kd_tree = KDTreeDataStructure(training_songs, training_genres)
    classifier = KnnSongClassifier(k, kd_tree)

    num_matches, confusion_matrix = classifier.test(test_songs, test_genres)
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


PREDICTION_DIRECTORY = 'song_data/test/'


def test_k_fold():
    lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    split_list = split_in_k(lst, 4)
    print(split_list)

    songs, genres = ['song1', 'song2', 'song3'], ['genre1', 'genre2', 'genre3']
    k_fold = 2

    split_songs = split_in_k(songs, k_fold)
    split_genres = split_in_k(genres, k_fold)

    print(split_songs)
    print(split_genres)

    for i in range(k_fold):

        # Concatenate all the k_fold - 1 lists

        lst1 = []
        for j in range(k_fold):
            if j != i:
                lst1.append(split_songs[j])
            else:
                lst1.append([])

        lst1 = sum(lst1, [])

        print(lst1)

        training_songs = sum([split_songs[j] if j != i else [] for j in range(k_fold)], [])
        training_genres = sum([split_genres[j] if j != i else [] for j in range(k_fold)], [])

        print('k = {}'.format(i))
        print(training_songs)
        print(training_genres)


def test_songs_gaussian_k_fold(k_fold):
    songs, genres = get_training_songs_genres()

    split_songs = split_in_k(songs, k_fold)
    split_genres = split_in_k(genres, k_fold)

    accuracies = []

    for i in range(k_fold):
        # Concatenate all the k_fold - 1 lists
        training_songs = sum([split_songs[j] if j != i else [] for j in range(k_fold)], [])
        training_genres = sum([split_genres[j] if j != i else [] for j in range(k_fold)], [])

        test_songs = split_songs[i]
        test_genres = split_genres[i]

        classifier = GaussianSongClassifier(training_songs, training_genres)

        num_matches, _ = classifier.test(test_songs, test_genres)
        num_test_songs = len(test_songs)

        accuracy = (num_matches / num_test_songs) * 100
        accuracies.append(accuracy)

        logging.info('[k-fold iteration #{}] Matched {} out of {} songs, accuracy: {}%'
                     .format(i, num_matches, num_test_songs, accuracy))

    logging.info('Average accuracy for Gaussian (k_fold={}): {}%'.format(k_fold, sum(accuracies) / len(accuracies)))


def test_songs_knn_k_fold(k, k_fold):
    songs, genres = get_training_songs_genres()

    split_songs = split_in_k(songs, k_fold)
    split_genres = split_in_k(genres, k_fold)

    accuracies = []

    for i in range(k_fold):
        # Concatenate all the k_fold - 1 lists
        training_songs = sum([split_songs[j] if j != i else [] for j in range(k_fold)], [])
        training_genres = sum([split_genres[j] if j != i else [] for j in range(k_fold)], [])

        test_songs = split_songs[i]
        test_genres = split_genres[i]

        classifier = KnnSongClassifier(k, training_songs, training_genres)

        num_matches, _ = classifier.test(test_songs, test_genres)
        num_test_songs = len(test_songs)

        accuracy = (num_matches / num_test_songs) * 100
        accuracies.append(accuracy)

        logging.info('[k-fold iteration #{}] Matched {} out of {} songs, accuracy: {}%'
                     .format(i, num_matches, num_test_songs, accuracy))

    logging.info('Average accuracy for kNN (k_fold={}, k={}): {}%'.format(k_fold, k, sum(accuracies) / len(accuracies)))


def test_songs_gaussian():
    songs, genres = get_training_songs_genres()

    training_songs, test_songs = split_in_two(songs)
    training_genres, test_genres = split_in_two(genres)

    classifier = GaussianSongClassifier(training_songs, training_genres)

    num_matches, confusion_matrix = classifier.test(test_songs, test_genres)
    confusion_matrix.save_to_csv('report/csv/confusion_gaussian.csv')
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


def test_songs_knn_simple(k):
    songs, genres = get_training_songs_genres()

    training_songs, test_songs = split_in_two(songs)
    training_genres, test_genres = split_in_two(genres)

    classifier = KnnSongClassifier(k, training_songs, training_genres, data_structure='simple')

    num_matches, confusion_matrix = classifier.test(test_songs, test_genres)
    confusion_matrix.save_to_csv('report/csv/confusion_knn_{}.csv'.format(k))
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


def test_songs_svm():
    songs, genres = get_training_songs_genres()

    training_songs, test_songs = split_in_two(songs)
    training_genres, test_genres = split_in_two(genres)

    classifier = SvmSongClassifier(training_songs, training_genres)

    num_matches, confusion_matrix = classifier.test(test_songs, test_genres)
    confusion_matrix.save_to_csv('report/csv/confusion_svm.csv')
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


def test_songs_naive_bayes():
    songs, genres = get_training_songs_genres()

    training_songs, test_songs = split_in_two(songs)
    training_genres, test_genres = split_in_two(genres)

    classifier = NaiveBayesSongClassifier(training_songs, training_genres)

    num_matches, confusion_matrix = classifier.test(test_songs, test_genres)
    confusion_matrix.save_to_csv('report/csv/confusion_bayes.csv')
    num_test_songs = len(test_songs)

    logging.info('Matched {} out of {} songs, accuracy: {}%'
                 .format(num_matches, num_test_songs, (num_matches / num_test_songs) * 100))


def test_songs_neural_network():
    songs, genres = get_training_songs_genres()

    training_songs, test_songs = split_in_two(songs)
    training_genres, test_genres = split_in_two(genres)

    classifier = NeuralNetworkSongClassifier(training_songs, training_genres)

    num_matches, confusion_matrix = classifier.test(test_songs, test_genres)
    confusion_matrix.save_to_csv('report/csv/confusion_neural.csv')
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


def predict_songs_neural():
    songs, genres = get_training_songs_genres()

    classifier = NeuralNetworkSongClassifier(songs, genres)

    classifier.predict_directory(PREDICTION_DIRECTORY, '{}test_labels_neural_net.csv'.format(DATA_DIRECTORY))


def predict_songs_svm():
    songs, genres = get_training_songs_genres()

    classifier = NeuralNetworkSongClassifier(songs, genres)

    classifier.predict_directory(PREDICTION_DIRECTORY, '{}test_labels_svm.csv'.format(DATA_DIRECTORY))


def predict_songs_gaussian_process():
    songs, genres = get_training_songs_genres()

    classifier = GaussianProcessSongClassifier(songs, genres)

    classifier.predict_directory(PREDICTION_DIRECTORY, '{}test_labels_gaussian_process.csv'.format(DATA_DIRECTORY))


def predict_songs_ada():
    songs, genres = get_training_songs_genres()

    classifier = AdaSongClassifier(songs, genres)

    classifier.predict_directory(PREDICTION_DIRECTORY, '{}test_labels_gaussian_ada.csv'.format(DATA_DIRECTORY))


def predict_songs_qda():
    songs, genres = get_training_songs_genres()

    classifier = QdaSongClassifier(songs, genres)

    classifier.predict_directory(PREDICTION_DIRECTORY, '{}test_labels_gaussian_qda.csv'.format(DATA_DIRECTORY))


if __name__ == '__main__':
    t = time.time()

    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO)

    # test_pandas()

    # classify_songs_gaussian()
    # test_songs_knn(3)

    test_songs_gaussian()

    # predict_songs_knn(3)

    # classify_gaussian()
    # classify_nearest_neighbor(5)
    # classify_nearest_neighbor_lsh(5)
    # classify_nearest_neighbor_kd_tree(5)
    # classify_nearest_neighbor_kd_tree_sk(5)
    # classify_nearest_neighbor_ball_tree(5)
    # classify_gaussian_kaggle()
    # test_lsh(5)

    test_k_fold()

    # Conclusion: sklearn KD tree performs the best
    # Rule of thumb: k = sqrt(N) where N is training examples, so k = 1171

    print('Total runtime: {}'.format(time.time() - t))
