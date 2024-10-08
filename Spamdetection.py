from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Sample dataset (emails and labels: 1 = spam, 0 = not spam)
emails = ["Get cheap loans now", "Important meeting tomorrow", "Win big prizes", "Your account details"]
labels = [1, 0, 1, 0]

# Convert emails into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X, labels)

# Test the model on a new email
test_email = ["Congratulations! You have won a free ticket"]
X_test = vectorizer.transform(test_email)
prediction = model.predict(X_test)

print(f"Email is {'spam' if prediction[0] == 1 else 'not spam'}")
