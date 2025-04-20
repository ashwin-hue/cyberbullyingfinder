# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Step 1: Data Preparation

# Load the CSV file
df = pd.read_csv(r"D:\csv_files\final_hateXplain.csv")

# Display the first few rows
print(df.head())

# Drop unnecessary columns and keep 'comments' and 'label'
df = df[['comment', 'label']]

# Encode the labels
label_mapping = {'normal': 0, 'offensive': 1, 'hatespeech': 2}
df['label'] = df['label'].map(label_mapping)

# Handle NaN values in the label column
df = df.dropna(subset=['label'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['label'], test_size=0.2, random_state=42)

# Step 2: Text Preprocessing

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 3: Model Building

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Get the unique classes in the training data
unique_classes = sorted(df['label'].unique())
class_names = [name for name, label in label_mapping.items() if label in unique_classes]

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=class_names, labels=unique_classes))

# Step 4: Save the Model and Vectorizer
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
