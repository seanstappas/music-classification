# Music Genre Classification
Classifies songs into genres based on feature vectors consisting of 12 features. This is assignment 2 of the ECSE-526 class, as described [here](http://www.cim.mcgill.ca/~jer/courses/ai/assignments/as2/).

## Dependencies

The program depends on the the [`pandas`](https://pandas.pydata.org/pandas-docs/stable/install.html) and [`numpy`](https://pypi.python.org/pypi/numpy) libraries to load and manipulate the song data, as well as the [`scikit-learn`](http://scikit-learn.org/stable/install.html) library if using the k-d tree for kNN or any classifiers besides Gaussian and kNN. All of these libraries can be installed via [`pip`](https://pip.pypa.io/en/stable/).

## Usage

### Example Commands

Here are some examples of running the program:

Training with a Gaussian classifier:
```
python main.py --classifier gaussian train
```

Training with a kNN classifier (k = 1) and using cross-validation with 10 subsets:
```
python main.py --classifier knn -k 1 train --k_fold 10
```

Predicting the genres of new songs using the Gaussian classifier:
```
python main.py --classifier gaussian predict
```

### Help

To run the program, use the command-line interface in `main.py`. To see the list of available commands, run the following:

```
python main.py --help
```

This will print the following:

```
usage: main.py [-h] [-c {knn,svm,nn,qda,gaussian,gpc,ada}]
               [-l {info,debug,critical,warn,error}] [-d DATA_PATH]
               [-k K_NEAREST] [-s {simple,kd_tree}]
               {train,predict} ...

Music Genre Classification.

positional arguments:
  {train,predict}
    train               Train with training data and test data.
    predict             Predict on new data. Saves the result to CSV in the data path.

optional arguments:
  -h, --help            show this help message and exit
  -c {knn,svm,nn,qda,gaussian,gpc,ada}, --classifier {knn,svm,nn,qda,gaussian,gpc,ada}
                        The classifier to use.
  -l {info,debug,critical,warn,error}, --logging_level {info,debug,critical,warn,error}
                        The logging level.
  -d DATA_PATH, --data_path DATA_PATH
                        The path containing the 'test' and 'training'
                        directories, as well as the 'labels' CSV.
  -k K_NEAREST, --k_nearest K_NEAREST
                        The number of nearest neighbours to use for kNN.
  -s {simple,kd_tree}, --knn_data_structure {simple,kd_tree}
                        The data structure to store previous examples for kNN.
```

To see help relating to the `train` argument, run the following:

```
python main.py train --help
```

This will print the following:

```
usage: main.py train [-h] [-f K_FOLD]

optional arguments:
  -h, --help            show this help message and exit
  -f K_FOLD, --k_fold K_FOLD
                        The number of partitions to use for k-fold cross-
                        validation.
```

The `predict` argument has no further optional arguments.

### Default Values

Here are the default values for all the optional arguments:

Argument | Default Value
--- | ---
`--classifier` | `gaussian`
`--logging_level` | `info`
`--data_path` | `song_data/`
`--k_nearest` | 1
`--knn_data_structure` | `kd_tree`
`--k_fold` | 10

## Code Organization

The code relating to all the different classifiers used can be found in the `classifiers.py` file.

## Report

The report (`report.pdf`) and all related files (tex, plots and CSV files) can be found in the `report` directory.
