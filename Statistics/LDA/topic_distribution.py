import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your combined dataset with the LDA results
data_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/final_combined_dataset_excluding_topic3.xlsx'
df = pd.read_excel(data_path)


# Calculate the average probability for each topic
topic_columns = ['topic_1', 'topic_2', 'topic_4']  # Exclude topic_3 if it was removed
average_probabilities = df[topic_columns].mean()

# Rename topics for better readability
topic_name_mapping = {
    'topic_1': 'Crime and Mystery',
    'topic_2': 'Reader Engagement and Enjoyment',
    'topic_4': 'Narrative and World-Building'
}
average_probabilities.index = average_probabilities.index.map(topic_name_mapping)

# Display the average probabilities
print("Average Probability for Each Topic:")
print(average_probabilities)

# Visualization: Horizontal Bar Chart of Topic Probability Distribution
plt.figure(figsize=(10, 6))
plt.barh(average_probabilities.index, average_probabilities.values, color='#FF6F61')  # Coral pastel color
plt.xlabel('Average Probability')
plt.ylabel('Topics')
plt.title('Average Topic Probability Distribution Across Reviews')
plt.gca().invert_yaxis()  # Optional: invert y-axis for a cleaner look
plt.tight_layout()
plt.show()