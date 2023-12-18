import pandas as pd
import numpy as np
import os
import utils
import matplotlib.pyplot as plt
import pickle

# Define labels for Positive and Negative reviews
aes = {"pos": "Positive Reviews", "neg": "Negative Reviews"}

for dataset in ["train", "test"]:
    # Check if the file for exploratory data analysis (EDA) exists
    if os.path.exists("data/eda_{}.pkl".format(dataset)):
        # Open the file with pickle if it exists
        with open("data/eda_{}.pkl".format(dataset), 'rb') as file:
            data = pickle.load(file)
    else:
        # If the file doesn't exist, perform EDA and save the DataFrame to a pickle file
        data_path = "data/{}".format(dataset)
        labels = ["pos", "neg"]

        # Initialize an empty DataFrame
        data_buffer = []

        for label in ["pos", "neg"]:
            for file_path in os.listdir("{}/{}".format(data_path, label)):
                id_movie = file_path.split("_")[0]
                rating_movie = int(file_path.split("_")[1].split(".")[0])

                with open("{}/{}/{}".format(data_path, label, file_path), 'r', encoding="utf8") as f:
                    file_content = utils.clean_review(f.read())
                    data_buffer.append([id_movie, file_content, rating_movie, label])

        data = pd.DataFrame(data_buffer, columns=['IdMovie', 'Review', 'Rating', 'Label'])

        # Store the DataFrame in a pickle file
        with open("data/eda_{}.pkl".format(dataset), 'wb') as file:
            pickle.dump(data, file)

    # Pie chart for the distribution of positive and negative reviews
    label_counts = data['Label'].value_counts()
    slices_labels = ['Positive Reviews', 'Negative Reviews']
    sizes = [label_counts.get('pos', 0), label_counts.get('neg', 0)]

    plt.pie(sizes, labels=None, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.legend(slices_labels, loc='lower right')
    plt.savefig("report/figures/{}_labels.pdf".format(dataset), bbox_inches="tight")
    plt.close()

    # Bar chart for the distribution of ratings
    rating_counts = data['Rating'].value_counts().sort_index()
    colormap = plt.cm.RdBu
    colors = colormap(np.linspace(0, 1, len(rating_counts)))

    plt.bar([str(rating) for rating in rating_counts.index], [count for count in rating_counts], color=colors)
    plt.xlabel('Rating')
    plt.ylabel('Movies')
    plt.savefig("report/figures/{}_ratings.pdf".format(dataset), bbox_inches="tight")
    plt.close()

    # Histograms for the distribution of review lengths
    def count_words(text):
        words = text.split()
        word_count = len(words)
        return word_count

    pos_lengths = list(map(lambda x: count_words(x), data.loc[data['Label'] == "pos", 'Review']))
    neg_lengths = list(map(lambda x: count_words(x), data.loc[data['Label'] == "neg", 'Review']))
    data_range = [min(min(pos_lengths), min(neg_lengths)), max(max(pos_lengths), max(neg_lengths))]
    num_bins = 50

    plt.hist(pos_lengths, bins=np.linspace(data_range[0], data_range[1], num_bins + 1), alpha=0.5, label=aes["pos"],
             edgecolor='black', linewidth=1.2)
    plt.hist(neg_lengths, bins=np.linspace(data_range[0], data_range[1], num_bins + 1), alpha=0.5, label=aes["neg"],
             edgecolor='black', linewidth=1.2)
    plt.xlabel('Reviews\' Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("report/figures/{}_lengths.pdf".format(dataset), bbox_inches="tight")
    plt.close()

    # Boxplot for the distribution of review lengths based on ratings
    lengths_ratings = []
    for i in sorted(list(set(data["Rating"]))):
        lengths_ratings.append(list(map(lambda x: count_words(x), data.loc[data['Rating'] == i, 'Review'])))

    plt.boxplot(lengths_ratings, labels=['1', '2', '3', '4', '7', '8', '9', '10'])
    plt.xlabel('Movies\' Rating')
    plt.ylabel('Reviews\' Length')
    plt.savefig("report/figures/{}_lengths_ratings.pdf".format(dataset), bbox_inches="tight")
    plt.close()