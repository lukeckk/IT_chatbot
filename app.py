from flask import Flask, render_template, request, jsonify
from chat import get_response
import requests
from bs4 import BeautifulSoup
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.get("/")
def index_get():
    url = "https://www.greenriver.edu/students/online-services/va.html"
    response = requests.get(url)
    contents = response.text

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(contents, "html.parser")

    # Select the second .card-body element
    card_bodies = soup.select(".card-body")
    second_card_body = card_bodies[1]

    return render_template("base.html", second_card_body=second_card_body)

@app.post("/predict")
def predict():
    text = request.get_json().get("message")       # message retrieves from msg1 in app.js
    # to do : check if text is valid
    response = get_response(text)
    message = {"answer": response}      # answer is also from js
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)



