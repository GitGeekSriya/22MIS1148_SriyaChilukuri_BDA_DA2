from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
frauds = []

# Step 1: Ensure you have the true and predicted labels
# True labels (from the dataset)
data = pd.read_csv('C:/Users/chsri/Desktop/fraudTest.csv')
true_labels = data['is_fraud'].values  # Assuming 'is_fraud' column contains the true labels

# Predicted fraud labels (1 = fraud, 0 = normal)
# If 'frauds' contains the indices of potential frauds:
predicted_frauds = [1 if i in frauds else 0 for i in range(len(true_labels))]

# Step 2: Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_frauds)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Step 3: Classification Report
class_report = classification_report(true_labels, predicted_frauds)

# Print the classification report
print("\nClassification Report:")
print(class_report)
