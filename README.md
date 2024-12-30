# IT Support Chatbot for Green River College

This project is an **NLP-powered chatbot** developed for the **Green River College IT Student Helpdesk Department**, designed to handle over 100 daily queries with **95% accuracy**. The chatbot leverages advanced NLP techniques and a **feed-forward neural network (FFNN)** for intent recognition, integrating tools for dynamic FAQ management and real-time updates to ensure response relevance.

---

## Live Demo
The chatbot is live and accessible at:  
👉 [Green River College IT Support Chatbot](https://grc-itchatbot.onrender.com/)

---

## Features

- **Intent Recognition**:
  - Utilizes a **feed-forward neural network (FFNN)** implemented in PyTorch for accurate classification of user intents.
  
- **NLP Preprocessing**:
  - Powered by **NLTK** for tokenization, stemming, and other text preprocessing tasks.
  
- **Dynamic FAQ Management**:
  - Integrated with the **Google Sheets API** to manage and update FAQs in real time.
  
- **Web Scraping for Updates**:
  - A custom web scraping system tracks changes on the college website to ensure chatbot responses remain up-to-date.

---

## Technologies Used

- **Python**: Core programming language.
- **Flask**: Backend framework for deploying the chatbot.
- **NLTK**: For natural language preprocessing.
- **PyTorch**: For implementing the feed-forward neural network.
- **Google Sheets API**: For dynamic FAQ management.
- **BeautifulSoup / Selenium**: For web scraping (if applicable).

---

## How It Works

1. **User Query**: The user submits a query to the chatbot interface.
2. **Preprocessing**:
   - The query is tokenized and preprocessed using NLTK.
3. **Intent Recognition**:
   - The processed input is fed into the FFNN, which classifies the user's intent.
4. **Dynamic FAQ Matching**:
   - The chatbot pulls responses dynamically from Google Sheets if applicable.
5. **Response Generation**:
   - The chatbot provides a relevant response to the user.
6. **Real-Time Updates**:
   - A web scraping system ensures responses reflect the latest updates from the college website.

---

## Setup and Installation

### Clone the Repository
```bash
git clone https://github.com/yourusername/it-support-chatbot.git
cd it-support-chatbot
