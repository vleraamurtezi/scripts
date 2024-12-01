import gensim
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load your best LDA model
best_model_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/best_lda_model.gensim'
best_model = gensim.models.LdaModel.load(best_model_path)

# Define topics, custom pastel color maps, and number of words per topic
num_words = 10  # Adjust if needed
topic_names = {
    0: 'Crime and Mystery',
    1: 'Reader Engagement and Enjoyment',
    3: 'Narrative and World-Building'
}
# Adjusted darker pastel colors: coral, yellow, green
darker_pastel_colors = {
    'Crime and Mystery': 'Oranges',      # Slightly darker coral pastel
    'Reader Engagement and Enjoyment': 'YlOrBr_r',  # Reversed yellow pastel for darker tones
    'Narrative and World-Building': 'BuGn'          # Darker green pastel
}

# Generate Word Clouds for Each Topic with Darker Pastel Colors and Print Words
for topic_id, topic_name in topic_names.items():
    # Get the top words for the current topic with probabilities
    words = best_model.show_topic(topic_id, topn=num_words)
    word_freq = {word: prob for word, prob in words}
    
    # Print the top words and probabilities in the terminal
    print(f"\nTop words for {topic_name} (Topic {topic_id}):")
    for word, prob in words:
        print(f"{word}: {prob:.4f}")

    # Generate word cloud with slightly darker pastel colors
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap=darker_pastel_colors[topic_name]).generate_from_frequencies(word_freq)
    
    # Display the word cloud without a title
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axes, including titles and labels
    plt.show()