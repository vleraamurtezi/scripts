import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with actual file path)
df = pd.read_excel('/Users/vleramurtezi/Desktop/Thesis/Data/LDA/filtered_topics_output_sequential.xlsx')

# Count sentiment categories within each genre for both VADER and RoBERTa
vader_genre_counts = df.groupby(['Genre', 'vader_sentiment']).size().unstack().fillna(0)
roberta_genre_counts = df.groupby(['Genre', 'roberta_sentiment']).size().unstack().fillna(0)

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
roberta_genre_counts.plot(kind='bar', stacked=True, colormap='Pastel1', ax=plt.gca())  # Changed colormap to Pastel1
plt.title('RoBERTa Sentiment Distribution Across Genres')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='RoBERTa Sentiment')
plt.tight_layout()
plt.show()

