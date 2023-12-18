import argparse
import os
import pickle
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from gensim.models import Word2Vec
from gensim.models.fasttext import FastText


# Function to define machine learning pipeline and hyperparameter grid
def executive_model(classifier):
    classifiers = {
        "svm": {"object": SVC(), "params": {"kernel": ['linear', 'rbf'], "C": [0.1, 1, 10], "gamma": [1, 'scale', 'auto']}},
        "lr": {"object": LogisticRegression(), "params": {"C": [0.1], "penalty": ['l2'], "max_iter": [1000], "solver": ["saga"]}}
    }

    # Create a machine learning pipeline
    pipeline = Pipeline([
        ('classifier', classifiers[classifier]["object"])
    ])

    # Define hyperparameter grid for GridSearchCV
    param_grid = {"classifier__{}".format(parameter): classifiers[classifier]["params"][parameter] for parameter in
                  classifiers[classifier]["params"]}

    return pipeline, param_grid


def average_word_vectors(model, words):
    """
    Given a Word2Vec model and a list of words, return the average word vector.
    """
    valid_words = [word for word in words if word in model.wv.key_to_index]

    if len(valid_words) == 0:
        # If none of the words are in the model's vocabulary, return a vector of zeros
        return [0.0] * model.vector_size

    # Calculate the average word vector
    avg_vector = sum(model.wv[word] for word in valid_words) / len(valid_words)

    return avg_vector


def average_word_vectors_for_column(model, dataframe, column_name):
    """
    Given a Word2Vec model and a pandas DataFrame with a column of lists of words,
    return a list of lists where each element is a vector representing the average of the embeddings of the words.
    """
    word_vectors_list = []

    for words_list in dataframe[column_name]:
        avg_vector = average_word_vectors(model, words_list)
        word_vectors_list.append(avg_vector)

    return word_vectors_list


# Input to decide Embedder and Model
parser = argparse.ArgumentParser(description="Text classification with pipeline and grid search.")
parser.add_argument("-e", type=str, default="w", choices=["w", "f"], help="Select \"w\" for Word2Vec, \"f\" for FastText.")
parser.add_argument("-c", type=str, default="svm", choices=["svm", "lr"],
                    help="Select \"svm\" for SVM, \"lr\" for Logistic Regression")
args = parser.parse_args()

# Load and process training set
with open('data/train.pkl', 'rb') as file:
    train_set = pickle.load(file)

# Load and process test set
with open('data/test.pkl', 'rb') as file:
    test_set = pickle.load(file)

# Create different models based on the number of features
models = {100: [], 300: [], 500: [], 1000: []}
for num_features in models:
    if args.e == "w":
        vectorizer = Word2Vec(sentences=train_set["words"], vector_size=num_features)
    else:
        vectorizer = FastText(sentences=train_set["words"], vector_size=num_features)
    models[num_features] = vectorizer

# Load and process a subset of the training set
with open('data/train_lite.pkl', 'rb') as file:
    train_subset = pickle.load(file)

# Perform grid search for different models and hyperparameters
classification_models = pd.DataFrame()
for num_features in models:
    kfold = KFold(n_splits=4, shuffle=True)
    pipeline, param_grid = executive_model(args.c)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring="f1", verbose=10)

    train_subset_embedded = average_word_vectors_for_column(models[num_features], train_subset, 'words')
    grid_search.fit(train_subset_embedded, train_subset["label"])

    model_results = pd.DataFrame(grid_search.cv_results_)
    model_results["max_features"] = num_features
    classification_models = pd.concat([classification_models, model_results], axis=0)

# Display and select the top k models based on mean test score
classification_models = classification_models.sort_values(by='mean_test_score', ascending=False)
k = 5
top_k_classification_models = classification_models.head(k)

# Find the best model among the top k models
best_f1_score = 0.0
for index, model_row in top_k_classification_models.iterrows():
    hyperparameters = model_row['params']
    model, _ = executive_model(args.c)
    model.set_params(**hyperparameters)

    train_set_embedded = average_word_vectors_for_column(models[num_features], train_set, 'words')
    model.fit(train_set_embedded, train_set["label"])

    test_set_embedded = average_word_vectors_for_column(models[num_features], test_set, 'words')
    predictions = model.predict(test_set_embedded)
    current_f1_score = f1_score(test_set["label"], predictions, average='weighted')

    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score

        best_model = {'model': model, 'predictions': predictions,
                      "classifier_params": hyperparameters, "embedder_params": model_row['max_features']
                      }

# Set current time for creating a unique directory name
curr_time = int(time.time())
# Create a directory for saving results
os.makedirs("report/results/word_emb__{}__{}__{}".format(args.e, args.c, curr_time))

# Open a file for writing performance results
with open("report/results/word_emb__{}__{}__{}/performance.tsv".format(args.e, args.c, curr_time), "w") as f:
    f.write("Embedder: {}\n".format(args.e))
    f.write("Embedder Size: {}\n".format(best_model["embedder_params"]))

    f.write("Classifier: {}\n".format(args.c))
    f.write("Classifier Params: {}\n".format(best_model["classifier_params"]))

    f.write("Accuracy: {}\n".format(accuracy_score(test_set["label"], predictions)))
    f.write("Precision: {}\n".format(precision_score(test_set["label"], predictions)))
    f.write("Recall: {}\n".format(recall_score(test_set["label"], predictions)))
    f.write("F1 Score: {}\n".format(f1_score(test_set["label"], predictions)))


# Save predictions to a pickle file
predictions_path = 'report/results/word_emb__{}__{}__{}/predictions.pkl'.format(args.e, args.c, curr_time)
with open(predictions_path, 'wb') as file:
    pickle.dump(best_model["predictions"], file)
