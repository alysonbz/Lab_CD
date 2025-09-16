from src.utils import load_fake_news_dataset
from sklearn.model_selection import train_test_split

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = load_fake_news_dataset()
# Create a series to store the labels: y
y = df['label']
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"],y,test_size = 0.3,random_state = 53)

# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = ____

# Transform the training data: tfidf_train
tfidf_train = ___

# Transform the test data: tfidf_test
tfidf_test =____

# Print selected features
print(____[5000:5100])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])
