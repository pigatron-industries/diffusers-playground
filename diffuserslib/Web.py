from .DiffusersWeb import DiffusersView
from flask import Flask


def startWebServer(ngrok_token = None):
    port = 5000
    app = Flask(__name__)
    DiffusersView.register(app)

    if (ngrok_token is not None):
        from pyngrok import ngrok
        ngrok.set_auth_token(ngrok_token)
        public_url = ngrok.connect(port).public_url
        app.config["BASE_URL"] = public_url
        print(" * ngrok tunnel \"{}\" -> \"http://127.0.0.1:{}\"".format(public_url, port))

    app.run()