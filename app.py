import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, recall_score
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load dataset
df = pd.read_csv('fake_job_postings.csv')

# Step 2: Combine and clean text
df = df[['title', 'location', 'description', 'requirements', 'fraudulent']]
df.fillna('', inplace=True)
df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['description'] + ' ' + df['requirements']
df = df[['text', 'fraudulent']]

# Step 3: TF-IDF Vectorization
X = df['text']
y = df['fraudulent']
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Step 4: Train-test split with stratify
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

print("Class distribution before SMOTE:")
print(pd.Series(y_train).value_counts())

# Step 5: Apply SMOTE on training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Step 6: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# Step 7: Evaluate model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nRecall (Fake Job Class):", recall_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 8: Save model and vectorizer for deployment
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
