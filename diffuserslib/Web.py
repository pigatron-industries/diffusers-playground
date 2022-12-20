from .DiffusersWeb import DiffusersView
from flask import Flask

def startWebServer():
    app = Flask(__name__)
    DiffusersView.register(app)
    app.run()