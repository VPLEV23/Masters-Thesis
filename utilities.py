import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from surprise import KNNBasic, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from surprise import KNNWithMeans, Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

# Drop any unnecessary columns
# Assuming 'timestamp' column is not needed for the recommendation system
ratings_df.drop(['timestamp'], axis=1, inplace=True)


# Check for missing values and handle them
# For instance, drop rows with missing values
ratings_df.dropna(inplace=True)
movies_df.dropna(inplace=True)


# Convert data types if necessary
# For instance, ensuring 'rating' is a float
ratings_df['rating'] = ratings_df['rating'].astype(float)


# Quick look at the data
print("Ratings DataFrame:")
print(ratings_df.head())
print("\nMovies DataFrame:")
print(movies_df.head())

plt.figure(figsize=(10, 5))
sns.countplot(x='rating', data=ratings_df)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

user_activity = ratings_df['userId'].value_counts()
plt.figure(figsize=(10, 5))
sns.histplot(user_activity, bins=50)
plt.title('User Activity')
plt.xlabel('Number of Ratings by User')
plt.ylabel('Frequency')
plt.show()


movie_popularity = ratings_df['movieId'].value_counts()
plt.figure(figsize=(10, 5))
sns.histplot(movie_popularity, bins=50)
plt.title('Movie Popularity')
plt.xlabel('Number of Ratings for Movie')
plt.ylabel('Frequency')
plt.show()

# Assuming 'genres' column in movies_df is pipe-separated
all_genres = [genre for sublist in movies_df['genres'].str.split('|').tolist() for genre in sublist]
genre_df = pd.DataFrame({'genre': all_genres})
plt.figure(figsize=(15, 7))
sns.countplot(y='genre', data=genre_df, order=genre_df['genre'].value_counts().index)
plt.title('Genre Distribution')
plt.xlabel('Count')
plt.ylabel('Genre')
plt.show()

# Create a 'Reader' to set the rating scale
reader = Reader(rating_scale=(1, 5))

# Create a 'Dataset' from the ratings dataframe
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Split the dataset into training and test sets
trainset, testset = train_test_split(data, test_size=0.2)

# Initialize a KNN model with user-based collaborative filtering
knn = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': True})

# Train the model
knn.fit(trainset)

# Make predictions on the test set
predictions = knn.test(testset)

# Evaluate the model
rmse = accuracy.rmse(predictions)

# Step 1: Feature Extraction
# Create a TF-IDF vectorizer and fit the 'genres' data
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

# Step 2: Compute Similarity
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Step 3: Make Recommendations
def recommend(movie_title):
    idx = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the scores of the 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# Example usage
recommended_movies = recommend('Toy Story (1995)')

# Collaborative Filtering using KNN
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
knn = KNNWithMeans(sim_options={'name': 'cosine', 'user_based': True})
knn.fit(trainset)

# Content-based Filtering using TF-IDF on genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])
cosine_sim = cosine_similarity(tfidf_matrix)

def hybrid_recommendations(user_id, movie_title):
    # Collaborative filtering
    idx_movie = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    movie_id = movies_df.iloc[idx_movie]['movieId']
    neighbors = knn.get_neighbors(movie_id, k=10)
    neighbor_movie_ids = [trainset.to_raw_iid(inner_id) for inner_id in neighbors]
    
    # Content-based filtering
    idx_content = movies_df.index[movies_df['title'] == movie_title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx_content]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    # Hybrid: combine and rank by average score
    combined_movies = list(set(neighbor_movie_ids + movie_indices))
    valid_movies = [movie for movie in combined_movies if movie in movies_df['movieId'].values]

    def score(movie):
        collaborative_score = knn.predict(user_id, movie).est
        content_score = cosine_sim[movies_df.index[movies_df['movieId'] == movie].tolist()[0], idx_content]
        return collaborative_score + content_score
    
    ranked_movies = sorted(valid_movies, key=score, reverse=True)
    
    return [movies_df[movies_df['movieId'] == movie]['title'].values[0] for movie in ranked_movies]

# Example
recommended_movies = hybrid_recommendations(1, 'Toy Story (1995)')
print(recommended_movies)