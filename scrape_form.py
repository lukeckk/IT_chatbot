import requests
from bs4 import BeautifulSoup


def scrape_form():
    # URL to scrape
    url = "https://www.greenriver.edu"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the form using its class name
        form = soup.find('form', {'class': 'd-flex nav-search'})

        if form:
            # Change the form's action attribute to point to the original site
            form['action'] = 'https://www.greenriver.edu/search.html'
            # Return the modified form HTML
            return form.prettify()
        else:
            return "Form not found."
    else:
        return f"Failed to retrieve the webpage. Status code: {response.status_code}"
