# Language Identification System

## Overview

This project implements a language identification system in Python. The goal of the system is to determine the language of a given text input.

The system currently supports the following languages:

* English
* German
* French
* Spanish
* Telugu

The project was developed using machine learning techniques and character-level text features.

---

## Approach

The implementation follows a simple text classification pipeline:

1. Load text samples from different language files.
2. Assign labels based on the language of each file.
3. Convert text into numerical features using TF-IDF.
4. Train machine learning models on the extracted features.
5. Predict the language of new text entered by the user.

To represent text, I used character-level n-grams with a range of 3 to 5 characters. Character n-grams are useful for language identification because different languages have unique character patterns and writing styles.

---

## Machine Learning Models

Two machine learning models were used:

### 1. Multinomial Naive Bayes

A lightweight probabilistic classifier commonly used for text classification tasks.

### 2. Logistic Regression

A widely used classification algorithm that performs well on sparse text features.

Both models were trained and evaluated on the dataset.

---

## Dataset

The dataset contains text samples from five languages:

* English
* German
* French
* Spanish
* Telugu

Each language is stored in a separate text file.

The dataset is automatically loaded and processed by the training script.

---



## How to Run

### Train the Model

Run the following command:

python src/train.py

This will:

* Load the dataset
* Extract TF-IDF features
* Train the models
* Evaluate performance
* Save the trained model and vectorizer

### Predict a Language

Run:

python src/predict.py

Enter any text when prompted, and the system will predict its language.

---

## Results

Current results on the available dataset:

* Multinomial Naive Bayes Accuracy: 82.5%
* Logistic Regression Accuracy: 82.5%

The model correctly identifies languages such as English, German, French, Spanish, and Telugu based on their character patterns.

---

## Libraries Used

* Python
* scikit-learn
* joblib

---

## References

The implementation uses standard machine learning and text classification techniques available in the scikit-learn documentation:

https://scikit-learn.org/

TF-IDF Vectorizer:
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

Multinomial Naive Bayes:
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

Logistic Regression:
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
