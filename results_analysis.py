import pickle
import matplotlib.pyplot as plt
import argparse
import os


# Input to decide Embedder and Model
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=str, default="t", choices=["t", "b", "w", "f"])
parser.add_argument("-c", type=str, default="svm", choices=["nb", "svm", "lr"])
args = parser.parse_args()


# Load and process the test set
with open('data/test.pkl', 'rb') as file:
    test_set = pickle.load(file)

# Add a processed_review column to the test set
test_set["processed_review"] = test_set['words'].apply(lambda x: ' '.join(x))

# Load predictions
results_path = 'report/results/doc_emb__t__lr__1702689014/' #Best Model among all the trials
with open(f'{results_path}predictions.pkl', 'rb') as file:
    predictions = pickle.load(file)

# Add predictions as a new column in the test set
test_set["predictions"] = predictions

# Plotting ratings
# Calculate and visualize the accuracy of predictions by rating
test_set['equal_labels_predictions'] = test_set['label'] == test_set['predictions']
grouped_data = test_set.groupby(['rating', 'equal_labels_predictions']).size().reset_index(name='count')
pivot_data = grouped_data.pivot(index='rating', columns='equal_labels_predictions', values='count').fillna(0)
pivot_data['fraction_equal'] = pivot_data[True] / (pivot_data[True] + pivot_data[False])

# Create a stacked bar plot
ax = pivot_data.drop(columns=['fraction_equal']).plot(kind='bar', stacked=True, colormap='RdBu')

# Annotate the plot with the fraction of observations with equal values
for i, (index, row) in enumerate(pivot_data.iterrows()):
    fraction = row['fraction_equal']
    ax.annotate(f'{fraction:.2%}', (i, row[True] + row[False]), ha='center', va='bottom')

# Customize the plot
plt.xlabel('Rating')
plt.ylabel('Movies')
plt.xticks(range(len(pivot_data.index)), pivot_data.index, rotation=0)
ax.get_legend().set_title(None)
legend_labels = ['Wrong Predictions', 'Correct Predictions']
ax.legend(legend_labels, loc='upper center')

# Save the plot
plt.savefig(f"{results_path}figures/t_lr_ratings.pdf", bbox_inches="tight")
plt.close()

# Plotting review lengths
# Create a stacked histogram of review lengths for correct and wrong predictions
plt.figure(figsize=(10, 6))
hist_data = [
    test_set[test_set['equal_labels_predictions'] == 1]['words'].apply(len),
    test_set[test_set['equal_labels_predictions'] == 0]['words'].apply(len)
]

# Plot the stacked histogram
n, bins, _ = plt.hist(hist_data, bins=30, stacked=True, alpha=0.5, label=['Correct Predictions', 'Wrong Predictions'], edgecolor='black')
proportions = [round((x/y)*100, 1) for x, y in zip(n[0], n[1])]

# Add numbers over each bar
for i in range(len(bins)-1):
    if n[0][i] > 50:
        plt.text((bins[i] + bins[i+1]) / 2, n[1][i], "{}%".format(proportions[i]), ha='center', va='bottom', fontsize=7)

# Add labels and title
plt.xlabel('Review\'s Length')
plt.ylabel('Frequency')
plt.legend()
plt.xlim((0, 800))

# Save the plot
plt.savefig(f"{results_path}figures/t_lr_lengths.pdf", bbox_inches="tight")
