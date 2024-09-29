import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load the Dataset (replace 'your_dataset.csv' with the actual file path)
data = pd.read_csv('C:/Users/chsri/Desktop/fraudTest.csv')

# Step 2: Explore the dataset (optional, you can comment out this part)
print(data.info())
print(data.describe())

# Step 3: Preprocess the Data
# Select relevant features for fraud detection
features = data[['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']]

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Step 4: Train the SOM
# Initialize SOM (10x10 grid, input length = number of features)
som = MiniSom(x=10, y=10, input_len=normalized_features.shape[1], sigma=0.3, learning_rate=0.5)

# Randomly initialize the SOM's weights
som.random_weights_init(normalized_features)

# Train the SOM for 100 iterations
som.train_random(normalized_features, 100)

# Step 5: Visualize the U-Matrix (distance map)
plt.figure(figsize=(10, 10))
plt.title('SOM U-Matrix (Distance Map)')

# U-Matrix shows distances between neighboring neurons
plt.pcolor(som.distance_map().T, cmap='coolwarm')
plt.colorbar()  # Show color scale

# Display the graph
plt.show()

# Step 6: Detect Potential Frauds
# Set a threshold for identifying fraud (you can adjust this value)
threshold = 0.6

# List to hold indices of potential frauds
frauds = []

# Loop through the dataset and check for anomalies
for i, x in enumerate(normalized_features):
    winner = som.winner(x)  # Get the winning neuron for each transaction
    # If the U-Matrix distance is above the threshold, mark as fraud
    if som.distance_map()[winner] > threshold:
        frauds.append(i)

print(f"Number of potential frauds detected: {len(frauds)}")

# Step 7: Evaluate the Model
# True labels for fraud detection (0 = normal, 1 = fraud)
true_labels = data['is_fraud'].values

# Create a predicted fraud list (1 for fraud, 0 for normal)
predicted_frauds = [1 if i in frauds else 0 for i in range(len(normalized_features))]

# Compute the confusion matrix and classification report
conf_matrix = confusion_matrix(true_labels, predicted_frauds)
print("Confusion Matrix:\n", conf_matrix)

# Print the classification report
print("\nClassification Report:\n", classification_report(true_labels, predicted_frauds))
