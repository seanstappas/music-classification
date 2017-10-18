from __future__ import division

import logging
import time

from classifiers import GaussianSongClassifier, KnnSongClassifier, SvmSongClassifier, NaiveBayesSongClassifier, \
    NeuralNetworkSongClassifier, GaussianProcessSongClassifier, AdaSongClassifier, QdaSongClassifier
from data_extractor import get_training_songs_genres, PREDICTION_DIRECTORY, DATA_DIRECTORY
from argparse import ArgumentParser

LOGGING_LEVELS = {
    'info': logging.INFO,
    'debug': logging.DEBUG,
    'warn': logging.WARN,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

CLASSIFIERS = {
    'gaussian': GaussianSongClassifier,
    'knn': KnnSongClassifier,
    'nn': NeuralNetworkSongClassifier,
    'svm': SvmSongClassifier,
    'gpc': GaussianProcessSongClassifier,
    'ada': AdaSongClassifier,
    'qda': QdaSongClassifier
}


def split_in_two(l, ratio=0.5):
    split_index = int(ratio * len(l))
    return l[:split_index], l[split_index:]


def split_in_k(l, k):
    split_index = len(l) // k
    return [l[split_index * i:split_index * i + split_index] if i < k - 1 else l[split_index * i:] for i in range(k)]


def train_k_fold(args):
    songs, genres = get_training_songs_genres(args.data_path)

    k_fold = args.k_fold
    classifier_name = args.classifier

    split_songs = split_in_k(songs, k_fold)
    split_genres = split_in_k(genres, k_fold)

    accuracies = []

    for i in range(k_fold):
        # Concatenate all the k_fold - 1 lists
        training_songs = sum([split_songs[j] if j != i else [] for j in range(k_fold)], [])
        training_genres = sum([split_genres[j] if j != i else [] for j in range(k_fold)], [])

        test_songs = split_songs[i]
        test_genres = split_genres[i]

        if classifier_name == 'knn':
            classifier = KnnSongClassifier(args.k_nearest, training_songs, training_genres,
                                           data_structure=args.knn_data_structure)
        else:
            classifier = CLASSIFIERS[classifier_name](training_songs, training_genres)

        num_matches, _ = classifier.test(test_songs, test_genres)
        num_test_songs = len(test_songs)

        accuracy = (num_matches / num_test_songs) * 100
        accuracies.append(accuracy)

        logging.info('[k-fold iteration #{}] Matched {} out of {} songs, accuracy: {}%'
                     .format(i, num_matches, num_test_songs, accuracy))

    logging.info('Average accuracy for {} (k_fold={}): {}%'.format(
        classifier_name, k_fold, sum(accuracies) / len(accuracies)))


def predict_songs(args):
    songs, genres = get_training_songs_genres(args.data_path)

    classifier_name = args.classifier

    if classifier_name == 'knn':
        classifier = KnnSongClassifier(args.k_nearest, songs, genres, data_structure=args.knn_data_structure)
    else:
        classifier = CLASSIFIERS[classifier_name](songs, genres)

    classifier.predict_directory(
        directory_name=args.data_path + 'test/',
        result_file_name='{}test_labels_{}.csv'.format(args.data_path, classifier_name))


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


def train(args):
    t = time.time()
    setup_logging(args.logging_level)
    train_k_fold(args)
    logging.info('Total runtime: {} s'.format(time.time() - t))


def predict(args):
    t = time.time()
    setup_logging(args.logging_level)
    predict_songs(args)
    logging.info('Total runtime: {} s'.format(time.time() - t))


def setup_logging(level):
    logging.basicConfig(
        format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s",
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=LOGGING_LEVELS[level])


def parse_command_line_arguments():
    parser = ArgumentParser(description='Music Genre Classification.')
    parser.add_argument('-c', '--classifier', default='gaussian', choices=CLASSIFIERS.keys(),
                        help='The classifier to use.')
    parser.add_argument('-l', '--logging_level', default='info', choices=LOGGING_LEVELS.keys(),
                        help='The logging level.')
    parser.add_argument('-d', '--data_path', default='song_data/',
                        help="The path containing the 'test' and 'training' directories, as well as the 'labels' CSV.")
    parser.add_argument('-k', '--k_nearest', type=int, default=1,
                        help="The number of nearest neighbours to use for kNN.")
    parser.add_argument('-s', '--knn_data_structure', default='kd_tree', choices=['simple', 'kd_tree'],
                        help="The data structure to store previous examples for kNN.")

    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train', help='Train with training data and test data.')
    parser_train.add_argument('-f', '--k_fold', type=int, default=10,
                              help="The number of partitions to use for k-fold cross-validation.")
    parser_train.set_defaults(func=train)

    parser_predict = subparsers.add_parser('predict', help='Predict on new data.')
    parser_predict.set_defaults(func=predict)

    args = parser.parse_args('-c knn -s simple -d song_data/ -k 5 predict'.split())
    # args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    parse_command_line_arguments()
