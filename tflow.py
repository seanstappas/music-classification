from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Data sets
import time

GENRE_TRAINING = "song_data/labels_train.csv"
GENRE_TEST = "song_data/labels_test.csv"

GENRES = ['classical', 'country', 'edm_dance', 'jazz', 'kids', 'latin', 'metal', 'pop', 'rnb', 'rock']


def main():

    # Load datasets.
    print('Load data')
    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=GENRE_TRAINING,
        target_dtype=np.int,
        features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
        filename=GENRE_TEST,
        target_dtype=np.int,
        features_dtype=np.float32)

    # Specify that all features have real-value data
    feature_columns = [tf.feature_column.numeric_column("x", shape=[12])]  # 12 features

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10],
                                            n_classes=10,  # 10 genres
                                            model_dir="/tmp/genre_model")
    # Define the training inputs
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(training_set.data)},
        y=np.array(training_set.target),
        num_epochs=None,
        shuffle=True)

    # Train model.
    print('Train model')
    classifier.train(input_fn=train_input_fn, steps=600000)

    # Define the test inputs
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": np.array(test_set.data)},
        y=np.array(test_set.target),
        num_epochs=1,
        shuffle=False)

    # Evaluate accuracy.
    print('Evaluate on test data')
    accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

    # Classify two new flower samples.
    new_samples = np.array(
        [[6.4, 3.2, 4.5, 1.5, 6.4, 3.2, 4.5, 1.5, 6.4, 3.2, 4.5, 1.5],
         [5.8, 3.1, 5.0, 1.7, 6.4, 3.2, 4.5, 1.5, 6.4, 3.2, 4.5, 1.5]], dtype=np.float32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": new_samples},
        num_epochs=1,
        shuffle=False)

    predictions = list(classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
        "New Samples, Class Predictions:    {}\n"
            .format(predicted_classes))


if __name__ == "__main__":
    t = time.time()
    main()
    print('Total runtime: {} s'.format(time.time() - t))
