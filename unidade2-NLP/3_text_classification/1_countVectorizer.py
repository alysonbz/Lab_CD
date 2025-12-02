
from src.utils import load_fake_news_dataset
# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

df = load_fake_news_dataset()

# Print the head of df
print(df.head())

# Create a series to store the labels: y
y = df['label']

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"],y,test_size = 0.3,random_state = 53)

# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer()

# # Transform the training data using X_train values  with fit_transform
count_train =

# Transform the test data using transform function
count_test = ____

# Print theselected features of the count_vectorizer
print(____[5000:5100])