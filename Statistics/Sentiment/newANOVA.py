# Import necessary libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load your dataset
data_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/final_combined_dataset_excluding_topic3.xlsx'
df = pd.read_excel(data_path)

# Step 1: Map categorical sentiment values to numeric for VADER and RoBERTa
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['vader_sentiment_num'] = df['vader_sentiment'].map(sentiment_mapping)
df['roberta_sentiment_num'] = df['roberta_sentiment'].map(sentiment_mapping)

# Ensure Genre and topic_categorical are strings or categorical
df['Genre'] = df['Genre'].astype(str)
df['topic_categorical'] = df['topic_categorical'].astype(str)

# Step 2: Check for missing values
print("Missing values in critical columns:")
print(df[['vader_sentiment_num', 'roberta_sentiment_num', 'Genre', 'topic_categorical']].isnull().sum())

# Drop rows with missing values if any
df = df.dropna(subset=['vader_sentiment_num', 'roberta_sentiment_num', 'Genre', 'topic_categorical'])

# Step 3: ANOVA for VADER Sentiment
print("\nRunning ANOVA for VADER Sentiment...")
model_vader = ols('vader_sentiment_num ~ C(Genre) * C(topic_categorical)', data=df).fit()
anova_table_vader = sm.stats.anova_lm(model_vader, typ=2)
print("\nANOVA Table for VADER Sentiment:")
print(anova_table_vader)

# Step 4: ANOVA for RoBERTa Sentiment
print("\nRunning ANOVA for RoBERTa Sentiment...")
model_roberta = ols('roberta_sentiment_num ~ C(Genre) * C(topic_categorical)', data=df).fit()
anova_table_roberta = sm.stats.anova_lm(model_roberta, typ=2)
print("\nANOVA Table for RoBERTa Sentiment:")
print(anova_table_roberta)

# Step 5: Post Hoc Test for Genre Main Effects (Tukey's HSD)
print("\nPost Hoc Test for Genre Main Effects (Tukey's HSD):")
# VADER Sentiment
tukey_vader_genre = pairwise_tukeyhsd(endog=df['vader_sentiment_num'], groups=df['Genre'], alpha=0.05)
print("\nTukey HSD Results for VADER Sentiment by Genre:")
print(tukey_vader_genre)

# RoBERTa Sentiment
tukey_roberta_genre = pairwise_tukeyhsd(endog=df['roberta_sentiment_num'], groups=df['Genre'], alpha=0.05)
print("\nTukey HSD Results for RoBERTa Sentiment by Genre:")
print(tukey_roberta_genre)

# Step 6: Post Hoc Test for Interaction Effects
print("\nPost Hoc Test for Interaction Effects:")

# Create interaction column
df['interaction'] = df['Genre'] + ' : ' + df['topic_categorical']

# VADER Interaction Post Hoc
mc_vader = MultiComparison(df['vader_sentiment_num'], df['interaction'])
posthoc_vader_interaction = mc_vader.tukeyhsd()
print("\nTukey HSD Results for VADER Sentiment Interaction Effects:")
print(posthoc_vader_interaction)

# RoBERTa Interaction Post Hoc
mc_roberta = MultiComparison(df['roberta_sentiment_num'], df['interaction'])
posthoc_roberta_interaction = mc_roberta.tukeyhsd()
print("\nTukey HSD Results for RoBERTa Sentiment Interaction Effects:")
print(posthoc_roberta_interaction)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the topic mapping
topic_rename_mapping = {
    "topic_1": "Crime and Mystery",
    "topic_2": "Reader Engagement and Enjoyment",
    "topic_4": "Narrative and World-Building"
}

# Map the topic_categorical column to descriptive names
df['topic_categorical_renamed'] = df['topic_categorical'].map(topic_rename_mapping)

# Group the data by Genre and Renamed Topic, calculating the mean sentiment scores
vader_heatmap_data = df.groupby(['Genre', 'topic_categorical_renamed'])['vader_sentiment_num'].mean().unstack()
roberta_heatmap_data = df.groupby(['Genre', 'topic_categorical_renamed'])['roberta_sentiment_num'].mean().unstack()

# Increase the figure size and adjust layout for VADER
plt.figure(figsize=(12, 8))
sns.heatmap(vader_heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Mean Sentiment Score'},
            linewidths=0.5)
#plt.title('VADER Sentiment Heatmap: Genre vs. Topics', fontsize=16)
plt.xlabel('Topics', fontsize=12)
plt.ylabel('Genres', fontsize=12)
plt.xticks(fontsize=10, rotation=45, ha='right')  # Rotate topic labels
plt.yticks(fontsize=10)
plt.tight_layout()  # Ensure everything fits in the frame
plt.show()

# Increase the figure size and adjust layout for RoBERTa
plt.figure(figsize=(12, 8))
sns.heatmap(roberta_heatmap_data, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Mean Sentiment Score'},
            linewidths=0.5)
#plt.title('RoBERTa Sentiment Heatmap: Genre vs. Topics', fontsize=16)
plt.xlabel('Topics', fontsize=12)
plt.ylabel('Genres', fontsize=12)
plt.xticks(fontsize=10, rotation=45, ha='right')  # Rotate topic labels
plt.yticks(fontsize=10)
plt.tight_layout()  # Ensure everything fits in the frame
plt.show()

