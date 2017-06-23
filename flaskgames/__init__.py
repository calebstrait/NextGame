from flask import Flask
app = Flask(__name__)
from flaskgames import views
app.secret_key = '2435#$5@#45#$5345'
