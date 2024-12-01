import matplotlib.pyplot as plt

# Define your topic numbers and coherence scores
topic_numbers = [5, 7, 10, 12, 15]
coherence_scores = [/* Add your saved coherence scores here */]

# Define topic names for each number of topics
topic_descriptions = {
    5: 'General Fiction, Reflection, Fantasy, Blogging, Classic Literature',
    7: 'General Fiction, Reflection, Fantasy, Blogging, Classic Literature, General Appeal, Crime',
    10: 'General Fiction, Reflection, Fantasy, Blogging, Classic Literature, General Appeal, Crime, Romance, Non-English Literature, Horror',
    12: 'Includes additional niche topics or genre-specific themes',
    15: 'Detailed topic breakdown with more specific genre evaluations'
}

# Plot coherence scores
plt.figure(figsize=(10, 6))
plt.plot(topic_numbers, coherence_scores, marker='o')
plt.title('Coherence Score by Topic Number')
plt.xlabel('Number of Topics')
plt.ylabel('Coherence Score')

# Annotate each point with its topic description
for i, txt in enumerate(topic_numbers):
    plt.annotate(topic_descriptions[txt], (topic_numbers[i], coherence_scores[i]), 
                 textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

plt.tight_layout()  # Ensures everything fits within the figure area
plt.show()
