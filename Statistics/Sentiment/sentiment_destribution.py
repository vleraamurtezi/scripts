import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace with actual file path or DataFrame)
df = pd.read_excel('/Users/vleramurtezi/Desktop/Thesis/Data/LDA/filtered_topics_output_sequential.xlsx')

# Calculate sentiment polarity counts for VADER
vader_counts = df['vader_sentiment'].value_counts()

# Display VADER sentiment polarity counts in a standalone figure with pastel coral color
plt.figure(figsize=(6, 6))
vader_counts.plot(kind='bar', color='#F7A399')  # Pastel coral color for VADER
plt.title('VADER Sentiment Polarity Counts')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Keep x labels horizontal for readability
plt.show()

# Calculate sentiment polarity counts for RoBERTa
roberta_counts = df['roberta_sentiment'].value_counts()

# Display RoBERTa sentiment polarity counts in a standalone figure with pastel coral color
plt.figure(figsize=(6, 6))
roberta_counts.plot(kind='bar', color='#F7A399')  # Pastel coral color for RoBERTa
plt.title('RoBERTa Sentiment Polarity Counts')
plt.xlabel('Sentiment Polarity')
plt.ylabel('Count')
plt.xticks(rotation=0)  # Keep x labels horizontal for readability
plt.show()

# 2. Agreement Analysis between VADER and RoBERTa (optional)
# Cross-tabulate the categorical sentiment labels to see agreement
agreement_df = pd.crosstab(df['vader_sentiment'], df['roberta_sentiment'], rownames=['VADER'], colnames=['RoBERTa'])
print("Agreement between VADER and RoBERTa Sentiment Labels:")
print(agreement_df)

# 3. Calculate Cohen's Kappa for agreement score (optional)
from sklearn.metrics import cohen_kappa_score
kappa_score = cohen_kappa_score(df['vader_sentiment'], df['roberta_sentiment'])
print(f"Cohen's Kappa Score: {kappa_score:.3f}")
# Define the agreement DataFrame based on your provided counts

# Display the agreement table as a straightforward figure for inclusion in your paper
plt.figure(figsize=(6, 4))
plt.table(cellText=agreement_df.values, 
          rowLabels=agreement_df.index, 
          colLabels=agreement_df.columns, 
          cellLoc='center', 
          loc='center')
plt.axis('off')
plt.title("Agreement between VADER and RoBERTa Sentiment Labels")
plt.show()

# Optional: Print the table in console for verification
print("Agreement between VADER and RoBERTa Sentiment Labels:")
print(agreement_df)