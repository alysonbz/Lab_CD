import pandas as pd
import matplotlib.pyplot as plt
from src.utils import get_count_vectorizer_fake_news_dataset,get_tfidf_fake_newes_dataset

# Import the necessary modules
from sklearn import  metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

count_train,count_test , count_vectorizer ,count_y_train, count_y_test = get_count_vectorizer_fake_news_dataset()


# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = ___

# Fit the classifier to the training data
___
# Create the predicted tags: pred
pred = ____

# Calculate the accuracy score: score
score = ____
print(score)

# Calculate the confusion matrix: cm
cm = confusion_matrix(__, ___)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,  display_labels=['FAKE', 'REAL'])
disp.plot()
plt.show()
