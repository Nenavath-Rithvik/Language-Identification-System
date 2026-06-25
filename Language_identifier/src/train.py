import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Data folder
DATA_FOLDER = "data/large_dataset"
texts = []
labels = []

# Read all language files
for file_name in os.listdir(DATA_FOLDER):

    if file_name.endswith(".txt"):

        language = file_name.replace(".txt", "")

        file_path = os.path.join(DATA_FOLDER, file_name)

        with open(file_path, "r", encoding="utf-8") as file:

            lines = file.readlines()

            for line in lines:

                line = line.strip()

                if line:

                    texts.append(line)
                    labels.append(language)

# Dataset information
print("Total Sentences:", len(texts))
print("Languages Found:", set(labels))

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5)
)

X = vectorizer.fit_transform(texts)

print("Feature Matrix Shape:", X.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=0.2,
    random_state=42
)

# Train model
model = MultinomialNB()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

# Logistic Regression

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)

print("\nLogistic Regression Accuracy:", lr_accuracy)


print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Test sentence

user_text = input("\nEnter a sentence: ")

test_sentence = [user_text]

test_features = vectorizer.transform(test_sentence)

prediction = model.predict(test_features)

print("\nInput Sentence:", test_sentence[0])
print("Predicted Language:", prediction[0])

# Save model and vectorizer

joblib.dump(model, "models/language_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel saved successfully!")