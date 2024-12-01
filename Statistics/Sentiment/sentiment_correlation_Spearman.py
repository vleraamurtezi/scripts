import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Load your dataset
file_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA/filtered_topics_output_sequential.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Calculate custom compound score for RoBERTa
df['RoBERTa_compound'] = df['RoBERTa_pos'] - df['RoBERTa_neg']

# Check for missing values in VADER_compound and RoBERTa_compound
print("Missing values before dropping:")
print(df[['VADER_compound', 'RoBERTa_compound']].isnull().sum())

# Drop rows with missing values in these columns
df = df.dropna(subset=['VADER_compound', 'RoBERTa_compound'])

# Check for constant values in the columns
unique_vader = df['VADER_compound'].nunique()
unique_roberta = df['RoBERTa_compound'].nunique()

if unique_vader > 1 and unique_roberta > 1:
    # Calculate Spearman's Rank Correlation
    spearman_corr, spearman_p = spearmanr(df['VADER_compound'], df['RoBERTa_compound'])
    print(f"Spearman Correlation: {spearman_corr:.2f}")
    print(f"P-value: {spearman_p:.4f}")

    # 1. Scatter Plot with Trend Line
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['VADER_compound'], y=df['RoBERTa_compound'])
    sns.regplot(x=df['VADER_compound'], y=df['RoBERTa_compound'], scatter=False, color="red", label="Trend Line")
    plt.title(f"Scatter Plot of VADER vs. RoBERTa Sentiment Scores\nSpearman Correlation = {spearman_corr:.2f}")
    plt.xlabel("VADER Compound Score")
    plt.ylabel("RoBERTa Compound Score")
    plt.legend()
    plt.show()

    # 2. Heatmap for Correlation Matrix
    # Create a DataFrame with the sentiment scores to calculate and visualize the correlation matrix
    corr_df = df[['VADER_compound', 'RoBERTa_compound']]
    corr_matrix = corr_df.corr(method='spearman')

    # Plot the heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, cbar_kws={'shrink': .8})
    plt.title("Spearman Correlation Matrix")
    plt.show()
else:
    print("Unable to compute correlation: one or both columns have constant values.")
