# app.py
from flask import render_template, url_for, flash, redirect
from app import app, db, login_manager
from app.forms import RegistrationForm
from app.models import User
from .models import User
@app.route('/')
@app.route('/index')
def index():
    return "Hello, World!"

@app.route("/register", methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)  # Use the set_password method
        db.session.add(user)
        db.session.commit()
        flash(f'Account created for {form.username.data}!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
@app.route("/login")
def login():
    return "Login Page"

