import string
from flask import Flask, render_template
import pandas as pd
import nltk
from nltk.corpus import stopwords
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer



nltk.download('punkt')
nltk.download('stopwords')
# Initialize Flask app
app = Flask(__name__)

# Load the trained KMeans model
kmeans_model = joblib.load('kmeans_model.pkl')

# Define route for clustering news articles
@app.route('/')
def cluster_news_articles():
    # Load news articles data
    df = pd.read_csv('NewsArticlesClustering.csv')

    # Preprocess text
    def preprocess_text(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        return ' '.join(tokens)

    df['processed_content'] = df['title'].apply(preprocess_text)

    # Extract features from text using TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_content'])

    # Predict clusters for news articles
    df['cluster'] = kmeans_model.predict(tfidf_matrix)

    # Prepare data for rendering HTML template
    clustered_articles = {}
    for cluster_id in range(kmeans_model.n_clusters):
        articles_in_cluster = df[df['cluster'] == cluster_id]['title'].tolist()
        clustered_articles[cluster_id] = articles_in_cluster

    # Render HTML template with clustered articles
    return render_template('index.html', clustered_articles=clustered_articles)

if __name__ == '__main__':
    app.run(debug=True)
