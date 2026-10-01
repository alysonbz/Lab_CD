
from src.utils import load_fake_news_dataset
# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

df = load_fake_news_dataset()

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df['label']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"],y,test_size = 0.3,random_state = 53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')

# # Transform the training data using X_train values  with fit_transform
count_train = count_vectorizer.fit_transform(X_train.values)

# Transform the test data using transform function
count_test = count_vectorizer.transform(X_test.values)

# Print the selected features of the count_vectorizer
print(count_vectorizer[5000:5100])