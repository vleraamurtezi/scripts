import pandas as pd

# Load the dataset
df = pd.read_excel('/Users/vleramurtezi/Desktop/Thesis/Data/Reviews_with_Relevant_Topic_Distributions.xlsx')

# Define the mapping for sentiment conversion
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

# Convert VADER and RoBERTa sentiment columns to numerical scores
df['vader_sentiment'] = df['vader_sentiment'].map(sentiment_mapping)
df['roberta_sentiment'] = df['roberta_sentiment'].map(sentiment_mapping)

# Save the updated dataframe back to Excel
df.to_excel('/Users/vleramurtezi/Desktop/Thesis/Data/Reviews_with_Numeric_Sentiment.xlsx', index=False)
