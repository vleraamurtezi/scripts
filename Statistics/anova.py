import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/final_combined_dataset_excluding_topic3.xlsx'
df = pd.read_excel(data_path)
print(df.dtypes)


# Map categorical sentiment values to numeric for both VADER and RoBERTa
sentiment_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
df['vader_sentiment_num'] = df['vader_sentiment'].map(sentiment_mapping)
df['roberta_sentiment_num'] = df['roberta_sentiment'].map(sentiment_mapping)

# Step 1: Define the ANOVA model with interaction for VADER
model_vader = ols('vader_sentiment_num ~ C(Genre) * C(topic_categorical)', data=df).fit()
anova_table_vader = sm.stats.anova_lm(model_vader, typ=2)
print("ANOVA Table for VADER Sentiment:")
print(anova_table_vader)

# Step 2: Define the ANOVA model with interaction for RoBERTa
model_roberta = ols('roberta_sentiment_num ~ C(Genre) * C(topic_categorical)', data=df).fit()
anova_table_roberta = sm.stats.anova_lm(model_roberta, typ=2)
print("ANOVA Table for RoBERTa Sentiment:")
print(anova_table_roberta)


