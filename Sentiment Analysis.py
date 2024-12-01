import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch

# Load the dataset from an .xls file
file_path = r'/Users/vleramurtezi/Desktop/Thesis/Data/Cleaned_Reviews.xlsx'  # Replace with the path to your .xls file
data = pd.read_excel(file_path)

# VADER Sentiment Analysis with additional checks for non-string data
def vader_sentiment_analysis(text):
    analyzer = SentimentIntensityAnalyzer()
    # Check if the text is a valid string
    if isinstance(text, str):
        sentiment = analyzer.polarity_scores(text)
        return sentiment['neg'], sentiment['neu'], sentiment['pos'], sentiment['compound']
    else:
        # Return NaNs if the review text is missing or invalid
        return float('nan'), float('nan'), float('nan'), float('nan')

# Apply VADER to each review with the enhanced NaN check
data[['VADER_neg', 'VADER_neu', 'VADER_pos', 'VADER_compound']] = data['Review'].apply(
    lambda x: pd.Series(vader_sentiment_analysis(x))
)

# RoBERTa Sentiment Analysis
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
roberta_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=True)

# RoBERTa Sentiment Analysis with additional checks for non-string data
def roberta_sentiment_analysis(text):
    # Check if the text is a valid string
    if isinstance(text, str):
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        output = model(**encoded_input)
        scores = torch.nn.functional.softmax(output.logits, dim=-1)
        # RoBERTa scores for each label (Negative, Neutral, Positive)
        neg_score = scores[0][0].item()
        neu_score = scores[0][1].item()
        pos_score = scores[0][2].item()
        return neg_score, neu_score, pos_score
    else:
        # Return NaNs if the review text is missing or invalid
        return float('nan'), float('nan'), float('nan')

# Apply RoBERTa to each review with the enhanced NaN check
data[['RoBERTa_neg', 'RoBERTa_neu', 'RoBERTa_pos']] = data['Review'].apply(
    lambda x: pd.Series(roberta_sentiment_analysis(x))
)

# Save the results to a new .xls file
output_path = '/Users/vleramurtezi/Desktop/Thesis/Data/sentiment_analysis_results.xlsx'  # Replace with your desired output path
data.to_excel(output_path, index=False)

print("Sentiment analysis complete. Results saved to", output_path)
