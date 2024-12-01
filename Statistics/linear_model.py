import pandas as pd
import statsmodels.api as sm

# Load your data
data_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/final_combined_dataset_excluding_topic3.xlsx'
df = pd.read_excel(data_path)

# Map sentiment labels to numeric values for both VADER and RoBERTa
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['vader_sentiment_num'] = df['vader_sentiment'].map(sentiment_mapping)
df['roberta_sentiment_num'] = df['roberta_sentiment'].map(sentiment_mapping)

# List of topics to include as predictors, explicitly excluding 'topic_categorical'
topic_columns = [col for col in df.columns if 'topic_' in col and col != 'topic_categorical']

# Ensure that all topic columns are numeric and fill missing values with 0
df[topic_columns] = df[topic_columns].apply(pd.to_numeric, errors='coerce').fillna(0)

# Loop through each unique genre in the dataset to build a separate model for each
for genre in df['Genre'].unique():
    # Filter the data for the current genre
    genre_df = df[df['Genre'] == genre]

    # Select the target variable (e.g., VADER or RoBERTa sentiment)
    y = genre_df['vader_sentiment_num']  # Replace with 'roberta_sentiment_num' if using RoBERTa sentiment

    # Select the predictor variables (topics)
    X = genre_df[topic_columns]
    
    # Add a constant to the model (for the intercept)
    X = sm.add_constant(X)

    # Check if X or y is empty after filtering
    if X.empty or y.empty:
        print(f"\nSkipping genre '{genre}' due to insufficient data.")
        continue

    # Build the linear model
    model = sm.OLS(y, X).fit()

    # Print the summary of the model for each genre
    print(f"\nLinear Model Summary for Genre: {genre}")
    print(model.summary())
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare DataFrame for coefficient plot
coefficients = []
errors = []
genres = []
topics = []

# Mapping for topic names
topic_name_mapping = {
    'topic_1': 'Crime and Mystery',
    'topic_2': 'Reader Engagement and Enjoyment',
    'topic_4': 'Narrative and World-Building'
}

# Collecting coefficients and standard errors
for genre in df['Genre'].unique():
    genre_df = df[df['Genre'] == genre]
    y = genre_df['vader_sentiment_num']
    X = genre_df[topic_columns]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    for topic in topic_columns:
        coefficients.append(model.params[topic])
        errors.append(model.bse[topic])
        genres.append(genre)
        # Apply renaming
        topics.append(topic_name_mapping.get(topic, topic))

# Create DataFrame for plotting
coef_df = pd.DataFrame({
    'Genre': genres,
    'Topic': topics,
    'Coefficient': coefficients,
    'Error': errors
})

# Plot grouped coefficient plot with error bars
plt.figure(figsize=(12, 8))
sns.pointplot(data=coef_df, x='Topic', y='Coefficient', hue='Genre', dodge=0.5, join=False, capsize=0.1, errwidth=1, ci=None)
plt.errorbar(
    x=coef_df['Topic'],
    y=coef_df['Coefficient'],
    yerr=coef_df['Error'],
    fmt='none',
    ecolor='black',
    capsize=5
)

# Title and labels
#plt.title("Coefficient Estimates for Topics Across Genres with Confidence Intervals")
plt.xlabel("Topics")
plt.ylabel("Coefficient Value")

# Adjust legend to be inside the figure
plt.legend(title="Genre", bbox_to_anchor=(1, 0.5), loc='center left', borderaxespad=0)

plt.tight_layout()  # Adjust layout to ensure everything fits well
plt.show()
