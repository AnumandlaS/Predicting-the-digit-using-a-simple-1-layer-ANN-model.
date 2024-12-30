import pandas as pd
#pandas library used to read file

from sklearn.feature_extraction.text import TfidfVectorizer
# TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer converts raw text into numerical values based on how important a word is in a document relative to its frequency in the corpus.

from sklearn.model_selection import train_test_split
# This is used to split the dataset into training and testing sets

from sklearn.linear_model import LogisticRegression
# It's a simple classification algorithm that tries to predict a binary outcome 

from sklearn.metrics import accuracy_score, classification_report
#These functions help evaluate the model's performance.

# Loading your dataset
# Assume df has two columns: 'text' and 'label' (1 for AI-generated, 0 for human-generated)
df = pd.read_csv('labelled_train_set.csv')

# Preprocess and vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['Article'])
y = df['Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred,zero_division=1))
#1.Accuracy = 0.77
#Precision: Out of the instances predicted as a specific class, how many were correctly predicted?
#Recall: Out of all the actual instances of a specific class, how many were correctly predicted?
#F1-score: The harmonic mean of precision and recall, balancing both metrics.
#Support: The number of actual occurrences of each class in the test set.
#2.Accuracy = 0.88