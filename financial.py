nformation gain is used to measure the 
effectiveness of a split.
3. Variance Reduction (for Regression): For regression tasks, Random Forest uses variance 
reduction to decide the best split.
4. Prediction:
o For classification: The final prediction is based on the majority vote from all 
decision trees.
o For regression: The final prediction is the average of all tree predictions
5. Out-of-Bag (OOB) Error: The OOB error is calculated as the prediction error on the 
samples that were not selected for training a particular tree (i.e., those that are not in the 
bootstrap sample). It is used to estimate the generalization error of the model without 
needing a separate validation dataset.
30
7.4 SAMPLE CODE
pip installscikit-learn 
import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, 
confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA # For dimensionality reduction (optional)
# Step 1: Load and Prepare Large Dataset
# Load the large dataset (replace 'large_dataset.csv' with the path to your actual dataset) 
# Use dask for large datasets if pandas becomes slow
# df = pd.read_csv('large_dataset.csv')
# Example of creating a large dummy dataset for illustration 
num_samples = 100000 # Simulate a large dataset
data = {
'transaction_id': np.arange(1, num_samples + 1),
'description': np.random.choice(['Payment', 'Refund', 'Withdrawal', 'Deposit'], num_samples), 
'amount': np.random.uniform(50, 500, num_samples), # Random transaction amounts 
'is_fraud': np.random.choice([0, 1], num_samples) # Fraudulent (1) or Non-Fraudulent (0)
}
df = pd.DataFrame(data)
# Step 2: Preprocess the Data (Vectorize text features)
X = df[['description', 'amount']] # Features: transaction description and amount 
y = df['is_fraud'] # Target: fraud status

# Convert text to numeric using CountVectorizer (convert descriptions into numerical data) 
vectorizer = CountVectorizer()
X_transformed = vectorizer.fit_transform(X['description'])
# Combine with 'amount' feature
X_combined = np.hstack((X_transformed.toarray(), X['amount'].values.reshape(-1, 1)))
# Step 3: Split Dataset into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)
# Step 4: Dimensionality Reduction (Optional, for large data handling) 
# Using PCA for reducing dimensions (Optional)
pca = PCA(n_components=50) # Reduce to 50 dimensions 
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
# Step 5: Train Random Forest Model
rf_model = RandomForestClassifier(n_jobs=-1) # Use all CPU cores for parallel processing 
rf_model.fit(X_train_pca, y_train)
# Step 6: Predictions using Random Forest 
rf_predictions = rf_model.predict(X_test_pca)
# Step 7: Rule-Based Model (Simple Example)
# For simplicity, Rule-Based is just checking specific keywords in description (for demonstration) 
def rule_based_model(df):
fraud_keywords = ['Withdrawal', 'Deposit'] # These are examples of rules 
predictions = []
for desc in df['description']:
if any(keyword in desc for keyword in fraud_keywords): 
predictions.append(1)
else:
predictions.append(0)
return predictions
rule_based_predictions = rule_based_model(df)
# Step 8: Evaluate Performance 
# Rule-based evaluation
rule_based_accuracy = accuracy_score(y, rule_based_predictions) 
rule_based_precision = precision_score(y, rule_based_predictions) 
rule_based_recall = recall_score(y, rule_based_predictions) 
rule_based_f1 = f1_score(y, rule_based_predictions)
# Random Forest evaluation
rf_accuracy = accuracy_score(y_test, rf_predictions) 
rf_precision = precision_score(y_test, rf_predictions) 
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
# Step 9: Confusion Matrix Visualization
rule_based_cm = confusion_matrix(y, rule_based_predictions) 
rf_cm = confusion_matrix(y_test, rf_predictions)
# Visualizing Confusion Matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
sns.heatmap(rule_based_cm, annot=True, fmt='d', cmap='Blues', ax=ax1) 
ax1.set_title('Rule-Based Confusion Matrix')
ax1.set_xlabel('Predicted') 
ax1.set_ylabel('Actual')
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax2) 
ax2.set_title('Random Forest Confusion Matrix') 
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')

plt.tight_layout() 
plt.show()
# Step 10: Compare Rule-Based vs Random Forest (Performance Metrics) 
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rule_based_metrics = [rule_based_accuracy, rule_based_precision, rule_based_recall, 
rule_based_f1]
rf_metrics = [rf_accuracy, rf_precision, rf_recall, rf_f1]
# Create a DataFrame for comparison
metrics_values = np.array([rule_based_metrics, rf_metrics]) 
metrics_names = ['Rule-Based', 'Random Forest']
# Set up the matplotlib figure
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.35
ind = np.arange(len(metrics)) # the x locations for the groups
ax.bar(ind - width/2, metrics_values[0], width, label='Rule-Based', color='skyblue') 
ax.bar(ind + width/2, metrics_values[1], width, label='Random Forest', color='salmon')
ax.set_xlabel('Metrics') 
ax.set_ylabel('Scores')
ax.set_title('Comparison of Rule-Based System vs Random Forest') 
ax.set_xticks(ind)
ax.set_xticklabels(metrics, rotation=45, ha="right") 
ax.legend()
plt.tight_layout() 
plt.show()