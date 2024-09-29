# Fraud Detection Using Self-Organizing Maps (SOM)

## Overview
This project implements a fraud detection system utilizing Self-Organizing Maps (SOM) to identify potential fraudulent transactions from a dataset. By analyzing transaction features, the model highlights anomalies that could indicate fraud.

## Files
- **SOM.py**: The main script that loads data, preprocesses it, trains the SOM, detects potential frauds, and evaluates the model.
- **ConfusionMatrix_EvaluationReport.py**: A script that calculates and outputs the confusion matrix and classification report for model evaluation.

## About the Dataset
The dataset used for this project consists of transaction records and includes the following features:
- **amt**: Transaction amount
- **lat**: Latitude of the transaction
- **long**: Longitude of the transaction
- **city_pop**: Population of the city
- **unix_time**: Timestamp of the transaction
- **merch_lat**: Latitude of the merchant
- **merch_long**: Longitude of the merchant
- **is_fraud**: Ground truth label indicating whether the transaction is fraudulent (1) or not (0)

## Execution Steps

### Software Requirements
- **Python**: Version 3.12.4
- **Required packages**:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - minisom

### Hardware Requirements
- Minimum of **8 GB RAM**
- At least **2 CPU cores**
- Sufficient storage for dataset (approx. **100 MB**)

### Installation
1. **Install Python**: Ensure Python 3.12.4 is installed on your system.
2. **Install required packages**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn minisom
## Execution

1. **Run the Main Script**  
   Execute the main script to perform the SOM analysis:

   ```bash
   python SOM.py
## Run the Evaluation Script

After running the main script, execute the evaluation script to generate the confusion matrix and classification report:

```bash
python ConfusionMatrix_EvaluationReport.py

## Output Explanation

After executing the scripts, you will capture the following results:

### Confusion Matrix
This matrix shows the model's performance in terms of correctly and incorrectly classified transactions:

- **True Negatives (TN)**: The number of legitimate transactions correctly identified as non-fraudulent.
- **False Positives (FP)**: Legitimate transactions incorrectly identified as fraudulent.
- **False Negatives (FN)**: Fraudulent transactions incorrectly identified as legitimate.
- **True Positives (TP)**: Fraudulent transactions correctly identified.

### Classification Report
This report provides detailed metrics on the model's performance, including:

- **Precision**: The proportion of positive identifications that were actually correct (TP / (TP + FP)). A higher precision indicates fewer false positives.
- **Recall**: The ability of the model to find all the relevant cases (TP / (TP + FN)). A higher recall means the model detects more actual fraud cases.
- **F1-score**: The harmonic mean of precision and recall, providing a single metric to evaluate the model’s performance.
- **Support**: The number of actual occurrences of each class in the specified dataset.

### Key Inferences
- The dataset consists of 555,719 entries with 23 columns.
- The dataset includes the following data types: 5 float columns, 6 integer columns, and 12 object (string) columns.
- Memory usage of the dataset is approximately 97.5 MB.
- The `is_fraud` column indicates that only 0.39% of the transactions are fraudulent, with 2,145 fraudulent cases out of 555,719 records. This shows the dataset is highly imbalanced.

### Key Statistics for Important Features
- **Transaction amount (amt)**: Various transaction amounts are recorded, with a wide range across different transactions.
- **Latitude (lat) and Longitude (long)**: The dataset captures geographic coordinates of the transactions and the merchant locations.
- **City population (city_pop)**: Represents the population of the city where the transaction took place.
- **Unix time (unix_time)**: Records the exact time of the transaction in Unix timestamp format.
- **Merchant latitude and longitude (merch_lat and merch_long)**: Similar to transaction coordinates, but for the merchant's location.

### Visualization Inferences
- **Color Representation**: The colors range from blue to red, where blue indicates smaller distances between neurons (indicating similar data points) and red indicates larger distances (indicating dissimilar data points).
- **Anomalies Detection**: The dark red areas represent regions with high mean distances to neighboring neurons. These regions are potential anomalies, which in the context of fraud detection, could indicate suspicious or fraudulent transactions.
- **Cluster Formation**: The blue areas suggest clusters of similar transactions, which are likely normal, non-fraudulent transactions.

In summary, the dark red squares among the lighter colors are the key areas of interest for identifying potential fraudulent activities. This visualization helps in pinpointing anomalies that might not be easily detectable in raw data.

### Output
![SOM_output](https://github.com/user-attachments/assets/a92a9cff-4763-423b-bdb0-3a405aecec2f)
