import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the dataset
file_path = '/Users/vleramurtezi/Desktop/Thesis/Data/sentiment_analysis_results.xlsx'  # Update to your file path
data = pd.read_excel(file_path)

# Define a function to plot scatter plots with linear and polynomial trend lines with NaN handling
def plot_with_trendlines(x, y, x_label, y_label, title):
    # Remove rows with NaN or infinite values in the columns
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x_clean, y=y_clean, color='blue', label='Data Points')
    
    # Linear trend line
    try:
        z_linear = np.polyfit(x_clean, y_clean, 1)
        p_linear = np.poly1d(z_linear)
        plt.plot(x_clean, p_linear(x_clean), "r--", label='Linear Fit')
    except np.linalg.LinAlgError:
        print(f"Linear fit failed for {title}")

    # Polynomial (quadratic) trend line
    try:
        z_poly = np.polyfit(x_clean, y_clean, 2)
        p_poly = np.poly1d(z_poly)
        plt.plot(x_clean, p_poly(x_clean), "g-", label='Polynomial Fit (Degree 2)')
    except np.linalg.LinAlgError:
        print(f"Polynomial fit failed for {title}")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

# Plot each pair of sentiment scores
plot_with_trendlines(data['VADER_neg'], data['RoBERTa_neg'], 'VADER Negative', 'RoBERTa Negative', 'VADER vs RoBERTa Negative Scores')
plot_with_trendlines(data['VADER_neu'], data['RoBERTa_neu'], 'VADER Neutral', 'RoBERTa Neutral', 'VADER vs RoBERTa Neutral Scores')
plot_with_trendlines(data['VADER_pos'], data['RoBERTa_pos'], 'VADER Positive', 'RoBERTa Positive', 'VADER vs RoBERTa Positive Scores')

# Optional: If you have created a synthetic RoBERTa compound score, you can compare it with VADER's compound score
# Example code if you have a synthetic RoBERTa compound column
# plot_with_trendlines(data['VADER_compound'], data['RoBERTa_compound'], 'VADER Compound', 'RoBERTa Compound', 'VADER vs RoBERTa Compound Scores')
