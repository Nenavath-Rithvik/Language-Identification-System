import joblib

# Load model

model = joblib.load("models/language_model.pkl")

# Load vectorizer

vectorizer = joblib.load("models/vectorizer.pkl")

# Get user input

user_text = input("Enter a sentence: ")

# Convert to features

features = vectorizer.transform([user_text])

# Predict

prediction = model.predict(features)

print("Predicted Language:", prediction[0])