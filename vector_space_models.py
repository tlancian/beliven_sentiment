import argparse
import os
import pickle
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Function to define the machine learning pipeline and hyperparameter grid
def executive_model(classifier):
    classifiers = {
        "nb": {"object": MultinomialNB(), "params": {'alpha': [0.1, 0.5, 1.0], 'fit_prior': [True, False]}},
        "svm": {"object": SVC(), "params": {"kernel": ['linear', 'rbf'], "C": [0.1, 1, 10], "gamma": [1, 'scale', 'auto']}},
        "lr": {"object": LogisticRegression(), "params": {"C": [0.1, 1, 10], "penalty": ['l1', 'l2'], "max_iter": [1000], "solver": ["saga"]}}
    }

    # Create a machine learning pipeline
    pipeline = Pipeline([
        ('classifier', classifiers[classifier]["object"])
    ])

    # Define a hyperparameter grid for GridSearchCV
    param_grid = {"classifier__{}".format(parameter): classifiers[classifier]["params"][parameter] for parameter in
                  classifiers[classifier]["params"]}

    return pipeline, param_grid

# Input to decide Embedder and Model
parser = argparse.ArgumentParser(description="Text classification with a pipeline and grid search.")
parser.add_argument("-e", type=str, default="t", choices=["t", "b"], help="Select \"t\" for TfIdf, \"b\" for Bag of Words")
parser.add_argument("-c", type=str, default="svm", choices=["nb", "svm", "lr"],
                    help="Select \"nb\" for Naive Bayes, \"svm\" for SVM, \"lr\" for Logistic Regression")
args = parser.parse_args()

# Load and process the training set
with open('data/train.pkl', 'rb') as file:
    train_set = pickle.load(file)

train_set["processed_review"] = train_set['words'].apply(lambda x: ' '.join(x))

# Load and process the test set
with open('data/test.pkl', 'rb') as file:
    test_set = pickle.load(file)

test_set["processed_review"] = test_set['words'].apply(lambda x: ' '.join(x))

# Create different models based on the number of features
models = {1000: [], 3000: [], 5000: [], 10000: []}
for num_features in models:
    if args.e == "t":
        vectorizer = TfidfVectorizer(max_features=num_features)
    else:
        vectorizer = CountVectorizer(max_features=num_features)
    vectorizer.fit(train_set["processed_review"])
    models[num_features] = vectorizer

# Load and process a subset of the training set
with open('data/train_lite.pkl', 'rb') as file:
    train_subset = pickle.load(file)

train_subset["processed_review"] = train_subset['words'].apply(lambda x: ' '.join(x))

# Perform grid search for different models and hyperparameters
classification_models = pd.DataFrame()
for num_features in models:
    kfold = KFold(n_splits=4, shuffle=True)
    pipeline, param_grid = executive_model(args.c)
    grid_search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring="f1", verbose=10)

    train_subset_embedded = models[num_features].transform(train_subset["processed_review"])
    grid_search.fit(train_subset_embedded, train_subset["label"])

    model_results = pd.DataFrame(grid_search.cv_results_)
    model_results["max_features"] = num_features
    classification_models = pd.concat([classification_models, model_results], axis=0)

# Display and select the top k models based on the mean test score
classification_models = classification_models.sort_values(by='mean_test_score', ascending=False)
k = 5
top_k_classification_models = classification_models.head(k)

# Find the best model among the top k models
best_f1_score = 0.0
for index, model_row in top_k_classification_models.iterrows():
    hyperparameters = model_row['params']
    model, _ = executive_model(args.c)
    model.set_params(**hyperparameters)

    train_set_embedded = models[model_row['max_features']].transform(train_set["processed_review"])
    model.fit(train_set_embedded, train_set["label"])

    test_set_embedded = models[model_row['max_features']].transform(test_set["processed_review"])
    predictions = model.predict(test_set_embedded)
    current_f1_score = f1_score(test_set["label"], predictions, average='weighted')

    if current_f1_score > best_f1_score:
        best_f1_score = current_f1_score

        best_model = {'model': model, 'predictions': predictions,
                      "classifier_params": hyperparameters, "embedder_params": model_row['max_features']
                      }



# Set the current time for creating a unique directory name
curr_time = int(time.time())

# Create a directory for saving results
os.makedirs("report/results/vsm__{}__{}__{}".format(args.e, args.c, curr_time))

# Open a file for writing performance results
with open("report/results/vsm__{}__{}__{}/performance.tsv".format(args.e, args.c, curr_time), "w") as f:

    f.write("Embedder: {}\n".format(args.e))
    f.write("Embedder Size: {}\n".format(best_model["embedder_params"]))

    f.write("Classifier: {}\n".format(args.c))
    f.write("Classifier Params: {}\n".format(best_model["classifier_params"]))

    f.write("Accuracy: {}\n".format(accuracy_score(test_set["label"], predictions)))
    f.write("Precision: {}\n".format(precision_score(test_set["label"], predictions)))
    f.write("Recall: {}\n".format(recall_score(test_set["label"], predictions)))
    f.write("F1 Score: {}\n".format(f1_score(test_set["label"], predictions)))


# Save predictions to a pickle file
predictions_path = 'report/results/vsm__{}__{}__{}/predictions.pkl'.format(args.e, args.c, curr_time)
with open(predictions_path, 'wb') as file:
    pickle.dump(best_model["predictions"], file)


if args.c == "lr":

    logreg_model = best_model["model"].named_steps['classifier']

    # Extract and print feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'Feature': models[best_model["embedder_params"]].get_feature_names_out(),
        'Coefficient': logreg_model.coef_[0]
    })

    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)

    with open("report/results/vsm__{}__{}__{}/feature_importance.tsv".format(args.e, args.c, curr_time), "w") as f:

        # Print top 20
        f.write("Top 20 Features:\n")
        f.write(feature_importance.head(20).to_string(index=False, header=False))
        f.write("\n\n\n")

        # Print bottom 20
        f.write("Top 20 Features:\n")
        f.write(feature_importance.tail(20).to_string(index=False, header=False))