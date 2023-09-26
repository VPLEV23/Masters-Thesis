import pandas as pd
from app import db, app
from app.models import DatasetUser, Book, Rating


ratings_df = pd.read_csv('./data/Ratings.csv')



# Populate the Ratings table
with app.app_context():
    for _, row in ratings_df.iterrows():
        rating = Rating(dataset_user_id=row['User-ID'], isbn=row['ISBN'], book_rating=row['Book-Rating'])
        db.session.add(rating)
    db.session.commit()

print("Data has been successfully added to the database!")
