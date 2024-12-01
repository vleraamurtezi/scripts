import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load your dataset
# Assuming your dataset file is named 'sentiment_data.xlsx' and is located in the same directory
df = pd.read_excel('/Users/vleramurtezi/Desktop/Thesis/Data/sentiment_analysis_results.xlsx')

# Define thresholds for categorical labeling for both VADER and RoBERTa
def categorize_vader(compound_score):
    """Categorize VADER compound score into Positive, Neutral, or Negative."""
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def categorize_roberta(row):
    """Categorize RoBERTa sentiment by selecting the label with the highest probability."""
    # Get the label with the highest probability
    if max(row['RoBERTa_pos'], row['RoBERTa_neu'], row['RoBERTa_neg']) == row['RoBERTa_pos']:
        return 'Positive'
    elif max(row['RoBERTa_pos'], row['RoBERTa_neu'], row['RoBERTa_neg']) == row['RoBERTa_neg']:
        return 'Negative'
    else:
        return 'Neutral'

# Apply categorization functions to create categorical labels
df['vader_sentiment'] = df['VADER_compound'].apply(categorize_vader)
df['roberta_sentiment'] = df.apply(categorize_roberta, axis=1)

# Calculate agreement using Cohen’s Kappa
kappa_score = cohen_kappa_score(df['vader_sentiment'], df['roberta_sentiment'])
print(f"Cohen's Kappa agreement between VADER and RoBERTa: {kappa_score:.2f}")

# Save the updated dataset with categorical labels for VADER and RoBERTa
#output_file = '/Users/vleramurtezi/Desktop/Thesis/Data/sentiment_data_with_categorization.xlsx'
#df.to_excel(output_file, index=False)
#print(f"Updated dataset saved to {output_file}")

