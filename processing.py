import os
import utils
import pandas as pd
import pickle
import argparse
import random

# Initialize an empty DataFrame
# This will be used to store the processed data
data = pd.DataFrame()

# Input to decide Embedder and Model
# Command-line argument parser for specifying dataset size
parser = argparse.ArgumentParser(description="Text classification with pipeline and grid search.")
parser.add_argument("--l", action='store_true', help="Number of observations with whom to create a lite version of the dataset")
args = parser.parse_args()

# Loop through both training and testing datasets
for dataset in ["train", "test"]:
    data_buffer = []  # Buffer to store data before creating the DataFrame

    # Loop through positive and negative labels
    for label in ["pos", "neg"]:
        files = os.listdir("data/{}/{}".format(dataset, label))

        # Set the number of observations based on the command-line argument
        if args.l:
            obs = 2000
            random.shuffle(files)
            name_file = "{}_lite.pkl".format(dataset)
        else:
            obs = 12500  # Adjusted to maintain the original total
            name_file = "{}.pkl".format(dataset)

        # Process each file in the current label's directory
        for file_path in files[:obs]:
            id_movie = file_path.split("_")[0]
            rating_movie = int(file_path.split("_")[1].split(".")[0])

            with open("data/{}/{}/{}".format(dataset, label, file_path), 'r', encoding="utf8") as f:
                # Read the entire content of the file into a string variable
                file_content = f.read()
                data_buffer.append([id_movie, file_content, rating_movie, label])

    # Create a DataFrame from the data buffer
    data = pd.DataFrame(data_buffer, columns=['id_movie', 'review', 'rating', 'label'])

    # Mapping dictionary to convert labels to numeric values
    mapping = {'pos': 1, 'neg': 0}
    data['label'] = data['label'].map(mapping)

    # Process the reviews using the 'text_cleaner' function from the 'utils' module
    reviews_processed = [utils.text_cleaner(review) for review in data["review"]]
    data["words"] = reviews_processed

    # Store the DataFrame in a pickle file
    with open("data/{}".format(name_file), 'wb') as file:
        pickle.dump(data, file)