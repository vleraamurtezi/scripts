import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Load the dataset
file_path = '/Users/vleramurtezi/Desktop/Thesis/Data/sentiment_analysis_results.xlsx'  # Update to your file path
data = pd.read_excel(file_path)

# Function to calculate and print Pearson and Spearman correlations
def calculate_correlations(data, vader_column, roberta_column):
    # Drop rows with NaN values in the columns of interest
    data_clean = data[[vader_column, roberta_column]].dropna()
    
    # Extract clean columns
    vader_scores = data_clean[vader_column]
    roberta_scores = data_clean[roberta_column]
    
    # Calculate Pearson correlation
    pearson_corr, pearson_p = pearsonr(vader_scores, roberta_scores)
    
    # Calculate Spearman correlation
    spearman_corr, spearman_p = spearmanr(vader_scores, roberta_scores)
    
    # Display results
    print(f"{vader_column} vs {roberta_column}:")
    print(f"  Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
    print(f"  Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print()

# Run the correlation calculations for each pair of VADER and RoBERTa scores
calculate_correlations(data, 'VADER_neg', 'RoBERTa_neg')
calculate_correlations(data, 'VADER_neu', 'RoBERTa_neu')
calculate_correlations(data, 'VADER_pos', 'RoBERTa_pos')

# Optional: If you created a synthetic compound score for RoBERTa, compare it with VADER's compound score
# Uncomment and use this line if you have a synthetic compound score for RoBERTa
# calculate_correlations(data, 'VADER_compound', 'RoBERTa_compound')
