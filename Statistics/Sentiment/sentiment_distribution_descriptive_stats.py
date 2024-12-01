import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with actual file path)
df = pd.read_excel('/Users/vleramurtezi/Desktop/Thesis/Data/LDA/filtered_topics_output_sequential.xlsx')

# Count sentiment categories within each genre for both VADER and RoBERTa
vader_genre_counts = df.groupby(['Genre', 'vader_sentiment']).size().unstack().fillna(0)
roberta_genre_counts = df.groupby(['Genre', 'roberta_sentiment']).size().unstack().fillna(0)

# Calculate proportions for each sentiment category within each genre for VADER
vader_genre_proportions = vader_genre_counts.div(vader_genre_counts.sum(axis=1), axis=0)
roberta_genre_proportions = roberta_genre_counts.div(roberta_genre_counts.sum(axis=1), axis=0)

# Display basic descriptive statistics for VADER and RoBERTa sentiment counts by genre
vader_stats = vader_genre_counts.describe()
roberta_stats = roberta_genre_counts.describe()

# Display to understand distribution within genres
print("VADER Sentiment Counts Summary by Genre:")
print(vader_stats)

print("\nRoBERTa Sentiment Counts Summary by Genre:")
print(roberta_stats)

# Display sentiment proportions to understand relative distribution within genres
print("\nVADER Sentiment Proportions by Genre:")
print(vader_genre_proportions)

print("\nRoBERTa Sentiment Proportions by Genre:")
print(roberta_genre_proportions)

# Plot VADER sentiment distribution across genres
plt.figure(figsize=(10, 6))
vader_genre_counts.plot(kind='bar', stacked=True, colormap='Pastel1', ax=plt.gca())
plt.title('VADER Sentiment Distribution Across Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='VADER Sentiment')
plt.tight_layout()
plt.show()

# Plot RoBERTa sentiment distribution across genres with the same style as VADER
plt.figure(figsize=(10, 6))
roberta_genre_counts.plot(kind='bar', stacked=True, colormap='Pastel1', ax=plt.gca())
plt.title('RoBERTa Sentiment Distribution Across Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='RoBERTa Sentiment')
plt.tight_layout()
plt.show()
