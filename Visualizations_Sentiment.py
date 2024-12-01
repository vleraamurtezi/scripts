import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset - adjust the path as necessary
df = pd.read_excel('/Users/vleramurtezi/Desktop/Thesis/Data/sentiment_data_with_categorization.xlsx')

# Create contingency tables for VADER and RoBERTa sentiment distributions by genre
vader_counts = pd.crosstab(df['Genre'], df['vader_sentiment'])
roberta_counts = pd.crosstab(df['Genre'], df['roberta_sentiment'])

# Normalize counts to percentages to show relative sentiment distribution by genre
vader_percentages = vader_counts.div(vader_counts.sum(axis=1), axis=0) * 100
roberta_percentages = roberta_counts.div(roberta_counts.sum(axis=1), axis=0) * 100

# Function to plot grouped bar chart for sentiment distribution by genre
def plot_sentiment_distribution(data, title):
    genres = data.index
    sentiments = data.columns

    # Position of each genre on the x-axis
    x = np.arange(len(genres))
    width = 0.25  # Width of each bar

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bars for each sentiment category
    for i, sentiment in enumerate(sentiments):
        ax.bar(x + i * width, data[sentiment], width, label=sentiment)

    # Adding labels and title
    ax.set_xlabel('Genre')
    ax.set_ylabel('Percentage')
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels(genres, rotation=45)
    ax.legend(title="Sentiment")

    plt.tight_layout()
    plt.show()

# Plot grouped bar charts for VADER and RoBERTa
plot_sentiment_distribution(vader_percentages, "Sentiment Distribution by Genre (VADER)")
plot_sentiment_distribution(roberta_percentages, "Sentiment Distribution by Genre (RoBERTa)")
