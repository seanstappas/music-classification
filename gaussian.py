import numpy as np

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


class GaussianSongClassifier:
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


class NaiveGaussianGenreModel:
    def __init__(self, genre, song_samples):
        self.genre = genre
        self.mean_vector = np.mean(song_samples, axis=0)
        self.covariance_matrix = np.cov(song_samples, rowvar=False)
        self.inv_covariance_matrix = np.linalg.inv(self.covariance_matrix)

    def compute_error(self, x):
        a = x - self.mean_vector
        return a.dot(self.inv_covariance_matrix).dot(a)  # * GENRE_WEIGHTS[self.genre]
