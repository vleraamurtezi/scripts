import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset - adjust path if necessary
df = pd.read_excel('/Users/vleramurtezi/Desktop/Thesis/Data/sentiment_data_with_categorization.xlsx')

# Function to perform and interpret Chi-square test for sentiment distribution across genres
def chi_square_test_by_genre(df, sentiment_column):
    # Create a contingency table of sentiment labels across genres
    contingency_table = pd.crosstab(df['Genre'], df[sentiment_column])

    # Perform Chi-square test of independence
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"\nChi-square test results for {sentiment_column}:")
    print(f"Chi-square statistic: {chi2:.2f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.2e}")  # Show the p-value in scientific notation

    # Interpret the results
    if p < 0.05:
        print(f"There is a statistically significant difference in {sentiment_column} sentiment distribution across genres.")
    else:
        print(f"There is no statistically significant difference in {sentiment_column} sentiment distribution across genres.")

# Run Chi-square tests for both VADER and RoBERTa sentiment columns
chi_square_test_by_genre(df, 'vader_sentiment')
chi_square_test_by_genre(df, 'roberta_sentiment')