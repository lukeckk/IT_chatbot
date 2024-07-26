import requests
from bs4 import BeautifulSoup

url = "https://www.greenriver.edu/va/"
response = requests.get(url)
contents = response.text

# Parse the HTML using BeautifulSoup
soup = BeautifulSoup(contents, "html.parser")

# Select the second .card-body element
card_bodies = soup.select(".card-body")
second_card_body = card_bodies[1]
print(second_card_body.prettify())