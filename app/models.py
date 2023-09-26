from app import db, bcrypt
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    # Method to set password
    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    # Method to check password
    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class DatasetUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # This will be User-ID from users.csv
    location = db.Column(db.String(200))
    age = db.Column(db.Integer)
    ratings = db.relationship('Rating', backref='dataset_user', lazy=True)

class Book(db.Model):
    isbn = db.Column(db.String(20), primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    author = db.Column(db.String(200))
    year_of_publication = db.Column(db.Integer)
    publisher = db.Column(db.String(200))
    image_url_s = db.Column(db.String(500))
    image_url_m = db.Column(db.String(500))
    image_url_l = db.Column(db.String(500))
    ratings = db.relationship('Rating', backref='book', lazy=True)

class Rating(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_user_id = db.Column(db.Integer, db.ForeignKey('dataset_user.id'), nullable=False)
    isbn = db.Column(db.String(20), db.ForeignKey('book.isbn'), nullable=False)
    book_rating = db.Column(db.Integer, nullable=False)