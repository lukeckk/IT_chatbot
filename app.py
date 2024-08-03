from flask import Flask, render_template, request, jsonify
from chat import get_response
import requests
from bs4 import BeautifulSoup
from flask_cors import CORS
import nltk
from scrape_form import scrape_form


app = Flask(__name__)
CORS(app)

nltk.download('punkt')

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

    form_html = scrape_form()

    return render_template("base.html", second_card_body=second_card_body, form_html=form_html)

@app.post("/predict")
def predict():
    try:
        text = request.get_json().get("message")
        app.logger.info(f"Received message: {text}")
        response = get_response(text)
        app.logger.info(f"Generated response: {response}")
        message = {"answer": response}
        return jsonify(message)
    except Exception as e:
        app.logger.error(f"Error in /predict: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)



