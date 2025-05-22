import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import joblib

# Step 1: Load dataset
df = pd.read_csv('fake_job_postings.csv')

# Step 2: Combine and clean
df = df[['title', 'location', 'description', 'requirements', 'fraudulent']]
df.fillna('', inplace=True)
df['text'] = df['title'] + ' ' + df['location'] + ' ' + df['description'] + ' ' + df['requirements']
df = df[['text', 'fraudulent']]

# Step 3: TF-IDF Vectorization
X = df['text']
y = df['fraudulent']
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Step 4: Train-test split BEFORE SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 5: Apply SMOTE on training data
print("Before SMOTE:", Counter(y_train))
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("After SMOTE:", Counter(y_train_resampled))

# Step 6: Define RandomForest and hyperparameters
rf = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': [None, 'balanced']
}

# Step 7: GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                           cv=3, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(X_train_resampled, y_train_resampled)

# Step 8: Best model
best_rf = grid_search.best_estimator_

# Step 9: Evaluate on test set
y_pred = best_rf.predict(X_test)
print("\n✅ Best Parameters:", grid_search.best_params_)
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred))

# Step 10: Save model and vectorizer
joblib.dump(best_rf, "rf_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("\n✅ Model and vectorizer saved.")
