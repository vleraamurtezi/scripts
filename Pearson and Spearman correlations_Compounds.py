import pandas as pd
from scipy.stats import pearsonr, spearmanr

# Load the dataset
file_path = '/Users/vleramurtezi/Desktop/Thesis/Data/sentiment_analysis_results.xlsx'  # Update to your file path
data = pd.read_excel(file_path)

# Calculate synthetic RoBERTa compound score
# Formula: RoBERTa Compound = (-1 * RoBERTa_neg) + (0 * RoBERTa_neu) + (1 * RoBERTa_pos)
data['RoBERTa_compound'] = (-1 * data['RoBERTa_neg']) + (1 * data['RoBERTa_pos'])

# Drop rows with NaN values in VADER and RoBERTa compound columns
data_clean = data[['VADER_compound', 'RoBERTa_compound']].dropna()

# Calculate Pearson and Spearman correlations between VADER compound and synthetic RoBERTa compound
pearson_corr, pearson_p = pearsonr(data_clean['VADER_compound'], data_clean['RoBERTa_compound'])
spearman_corr, spearman_p = spearmanr(data_clean['VADER_compound'], data_clean['RoBERTa_compound'])

# Display results
print("VADER_compound vs RoBERTa_compound:")
print(f"  Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")
print(f"  Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
