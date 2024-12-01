import pandas as pd
import matplotlib.pyplot as plt

# Load your combined dataset with the LDA results
data_path = '/Users/vleramurtezi/Desktop/Thesis/Data/LDA FINAL/final_combined_dataset_excluding_topic3.xlsx'
df = pd.read_excel(data_path)


# Map topic names for visualization
topic_name_mapping = {
    'topic_1': 'Crime and Mystery',
    'topic_2': 'Reader Engagement and Enjoyment',
    'topic_4': 'Narrative and World-Building'
}
df['topic_categorical_named'] = df['topic_categorical'].map(topic_name_mapping)

# Calculate Topic Prevalence
topic_counts = df['topic_categorical_named'].value_counts()
topic_percentages = (topic_counts / len(df)) * 100

# Combine counts and percentages into a DataFrame for easy viewing
topic_prevalence_df = pd.DataFrame({
    'Topic': topic_counts.index,
    'Count': topic_counts.values,
    'Percentage': topic_percentages.values
})

# Display the topic prevalence data
print("Topic Prevalence (Counts and Percentages):")
print(topic_prevalence_df)

# Visualization: Bar Chart of Topic Prevalence with Coral Pastel Color
plt.figure(figsize=(12, 6))
plt.bar(topic_prevalence_df['Topic'], topic_prevalence_df['Percentage'], color='#FF6F61')  # Coral pastel color
plt.xlabel('Topics')
plt.ylabel('Percentage of Reviews')
plt.title('Topic Prevalence in Reviews')
plt.xticks(rotation=45, ha='right', fontsize=10, wrap=True)  # Rotate and wrap text
plt.tight_layout()  # Adjust layout to fit labels
plt.show()

# Pie Chart of Topic Prevalence
plt.figure(figsize=(8, 8))
plt.pie(
    topic_percentages, 
    labels=topic_counts.index, 
    autopct='%1.1f%%', 
    startangle=140, 
    colors=['#FF6F61', '#FFB6B9', '#FFDAC1']  # Coral pastel and related shades
)
plt.title('Topic Prevalence in Reviews')
plt.show()