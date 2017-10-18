import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from data_extractor import get_one_song_per_genre, get_songs_genre, get_songs_genres

GENRES = ['classical', 'country', 'edm_dance', 'jazz', 'kids', 'latin', 'metal', 'pop', 'rnb', 'rock']


def plot_two_feature_vectors_one_song_per_genre(feature1=0, feature2=1):
    genres_to_songs = get_one_song_per_genre()
    genre_xs = {}
    genre_ys = {}
    for genre in GENRES:
        genre_xs[genre] = []
        genre_ys[genre] = []

    for song, genre in zip(genres_to_songs.values(), genres_to_songs.keys()):
        for feature_vector in song:
            genre_xs[genre].append(feature_vector[feature1])
            genre_ys[genre].append(feature_vector[feature2])

    f = plt.figure()
    for genre in GENRES:
        plt.scatter(genre_xs[genre], genre_ys[genre], 2, label=genre)
    plt.xlabel('Feature {}'.format(feature1))
    plt.ylabel('Feature {}'.format(feature2))
    plt.legend()
    f.savefig('report/plots/two_feature_vectors_{}{}.pdf'.format(feature1, feature2), bbox_inches='tight')


def plot_two_feature_vectors_genre(feature1=0, feature2=1, genre='classical', num_songs=10):
    songs = get_songs_genre(genre, num_songs)
    genre_xs = []
    genre_ys = []

    for song in songs:
        for feature_vector in song:
            genre_xs.append(feature_vector[feature1])
            genre_ys.append(feature_vector[feature2])

    f = plt.figure()
    for i, song in enumerate(songs):
        plt.scatter(song[:, feature1], song[:, feature2], 2, label='{}_{}'.format(genre, i))
    plt.xlabel('Feature {}'.format(feature1))
    plt.ylabel('Feature {}'.format(feature2))
    plt.legend()
    f.savefig('report/plots/two_feature_vectors_{}_{}{}.pdf'.format(genre, feature1, feature2), bbox_inches='tight')


def plot_two_feature_vectors_genres(feature1=0, feature2=1, desired_genres=('classical', 'rnb'), num_songs=5):
    genres_to_songs = get_songs_genres(desired_genres, num_songs)
    genre_xs = {}
    genre_ys = {}
    for genre in desired_genres:
        genre_xs[genre] = []
        genre_ys[genre] = []

    for songs, genre in zip(genres_to_songs.values(), genres_to_songs.keys()):
        for song in songs:
            for feature_vector in song:
                genre_xs[genre].append(feature_vector[feature1])
                genre_ys[genre].append(feature_vector[feature2])

    f = plt.figure()
    for genre in desired_genres:
        plt.scatter(genre_xs[genre], genre_ys[genre], 2, label=genre)
    plt.xlabel('Feature {}'.format(feature1))
    plt.ylabel('Feature {}'.format(feature2))
    plt.legend()
    f.savefig('report/plots/two_feature_vectors_{}{}.pdf'.format(feature1, feature2), bbox_inches='tight')


def plot_all_combinations_of_two_feature_vectors_10_songs():
    for i in range(10):
        for j in range(i + 1, 10):
            plot_two_feature_vectors_one_song_per_genre(i, j)


def plot_all_combinations_of_two_feature_vectors_1_genre(genre):
    for i in range(10):
        for j in range(i + 1, 10):
            plot_two_feature_vectors_genre(i, j, genre)


def plot_all_combinations_2_vectors_all_genres():
    for genre in GENRES:
        if genre != 'classical' and genre != 'rnb':
            plot_all_combinations_of_two_feature_vectors_1_genre(genre)


def plot_all_combinations_2_vectors_2_genres():
    for i in range(10):
        for j in range(i + 1, 10):
            plot_two_feature_vectors_genres(i, j)


def plot_knn_accuracy(accuracies, label):
    f = plt.figure()
    ax = f.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    k_values = [k for k in range(1, len(accuracies) + 1)]
    plt.plot(k_values, accuracies, 'o-', label=label)
    plt.grid()
    plt.xlabel('k')
    plt.ylabel('Accuracy (%)')
    f.savefig('report/plots/knn_accuracy_{}.pdf'.format(label), bbox_inches='tight')


def plot_knn_iterations():
    test_accuracies = [
        60.0901882189,
        59.8902056466,
        60.0217845939,
        59.4266294876,
        59.3608400139,
        59.4937260369,
        59.3612757058,
        59.0297141861,
        58.7003311258,
        58.502091321,
    ]

    plot_knn_accuracy(test_accuracies, 'training')

    kaggle_accuracies = [
        57.786,
        56.557,
        56.557
    ]

    plot_knn_accuracy(kaggle_accuracies, 'kaggle')


if __name__ == '__main__':
    # plot_all_combinations_of_two_feature_vectors_10_songs()
    # plot_all_combinations_2_vectors_all_genres()
    # plot_all_combinations_2_vectors_2_genres()
    plot_knn_iterations()
