import logging

import pandas as pd

DATA_DIRECTORY = 'song_data/'
TRAINING_DIRECTORY = DATA_DIRECTORY + 'training/'
PREDICTION_DIRECTORY = DATA_DIRECTORY + 'test/'


def get_training_songs_genres(data_path=DATA_DIRECTORY, num_songs=None):
    """
    Get the training and test songs from the specified data path.

    :param data_path: the path to the test and training data
    :param num_songs: the maximum number of songs to return
    :return: the songs and genres
    """
    logging.info('Building training and test set.')
    labels_genres = pd.read_csv('{}labels.csv'.format(data_path))
    song_ids = labels_genres['id'].values
    songs = []
    for i, song_id in enumerate(song_ids):
        if num_songs is not None and i >= num_songs:
            break
        song = pd.read_csv('{}{}'.format(TRAINING_DIRECTORY, song_id)).values
        songs.append(song)
    genres = labels_genres['category'].values.tolist()
    return songs, genres


def get_one_song_per_genre():
    logging.info('Building training and test set.')
    labels_genres = pd.read_csv('{}labels.csv'.format(DATA_DIRECTORY))
    song_ids = labels_genres['id'].values
    genres = labels_genres['category'].values
    genres_to_songs = {}
    for song_id, genre in zip(song_ids, genres):
        if len(genres_to_songs) >= 10:
            break
        song = pd.read_csv('{}{}'.format(TRAINING_DIRECTORY, song_id)).values
        genres_to_songs[genre] = song
    return genres_to_songs


def get_songs_genre(desired_genre='classical', num_songs=1):
    logging.info('Building training and test set.')
    labels_genres = pd.read_csv('{}labels.csv'.format(DATA_DIRECTORY))
    song_ids = labels_genres['id'].values
    genres = labels_genres['category'].values
    songs = []
    for song_id, genre in zip(song_ids, genres):
        if len(songs) >= num_songs:
            break
        if genre == desired_genre:
            song = pd.read_csv('{}{}'.format(TRAINING_DIRECTORY, song_id)).values
            songs.append(song)
    return songs


def get_songs_genres(desired_genres=('latin', 'rnb'), num_songs_per_genre=10):
    logging.info('Building training and test set.')
    labels_genres = pd.read_csv('{}labels.csv'.format(DATA_DIRECTORY))
    song_ids = labels_genres['id'].values
    genres = labels_genres['category'].values
    genres_to_songs = {}
    song_nums = {}
    for g in desired_genres:
        genres_to_songs[g] = []
        song_nums[g] = 0
    for song_id, genre in zip(song_ids, genres):
        if genre in desired_genres:
            if song_nums[genre] < num_songs_per_genre:
                song = pd.read_csv('{}{}'.format(TRAINING_DIRECTORY, song_id)).values
                genres_to_songs[genre].append(song)
                song_nums[genre] += 1
    return genres_to_songs
