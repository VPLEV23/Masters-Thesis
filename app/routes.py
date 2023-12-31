from flask import render_template, url_for, flash, redirect, request
from app import app, db, login_manager, bcrypt
from app.forms import RegistrationForm
from app.models import User, Book
from .models import User
from flask_login import login_user, current_user,logout_user, login_required
from app.forms import LoginForm

@app.route('/')
@app.route('/index')
def index():
    return render_template('welcome.html')


@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user_by_username = User.query.filter_by(username=form.username.data).first()
        user_by_email = User.query.filter_by(email=form.email.data).first()

        if user_by_username:
            form.username.errors.append("That username is taken. Please choose a different one.")
        if user_by_email:
            form.email.errors.append("That email is already registered. Please choose a different one or log in.")

        if not user_by_username and not user_by_email:
            user = User(username=form.username.data, email=form.email.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash(f'Account created for {form.username.data}!', 'success')
            return redirect(url_for('login'))

    return render_template('register.html', title='Register', form=form)




@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user, remember=form.remember.data)
            # You can redirect to a dashboard or some other page after successful login
            return redirect(url_for('dashboard'))
        else:
            # If password check fails, show an error
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET'])
@login_required  # Assuming you have a login_required decorator
def dashboard():
    page = request.args.get('page', 1, type=int)
    books = Book.query.paginate(page=page, per_page=25)
    return render_template('dashboard.html', books=books)