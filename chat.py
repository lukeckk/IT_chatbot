import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # tell pytorch to use cuda is available, else use cpu

with open('dataset.json', 'r') as file:
    dataset = json.load(file)

FILE = "data.pth"
data = torch.load(FILE) # open the pytorch file

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

now = datetime.now()
pacific = pytz.timezone('America/Los_Angeles')
now_pacific = datetime.now(pacific)
date = now_pacific.strftime("%Y-%m-%d")

scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
credentials = ServiceAccountCredentials.from_json_keyfile_name("ss_credentials.json", scope)
client = gspread.authorize(credentials) # client email
# open the sheet
sheet = client.open("IT Helper Test").sheet1
header = sheet.row_values(1)    # get all values in the first row
# issue_column = sheet.acell('A1').value-
issue_column = header.index("Issue") + 1
issue_values = sheet.col_values(issue_column)
number_column = header.index("Number") + 1
number_values = sheet.col_values(number_column)
date_column = header.index("Date") + 1
date_values = sheet.col_values(date_column)


# name the bot
bot_name = "IT Helper"

def get_response(msg):
    sentence = tokenize(msg)   #tokenize user's input
    x = bag_of_words(sentence, all_words)   # use bag_of_Words for sentence and all_words
    x = x.reshape(1, x.shape[0])    # reshape it to 1 row, 0 column
    x = torch.from_numpy(x)      # covert  type because bag_of_words returns numpy array

    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]    #predicted.item() is the number, tag is the actual tag in dataset

    # check the probability to indentify tag
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for data in dataset["intents"]:
            if tag == data["tag"]:
                new_issue = data["tag"]
                found = False
                for row_index, issue in enumerate(issue_values):
                    if issue == new_issue and issue != "greeting" and issue != "goodbye" and sheet.cell(row_index + 1, date_column).value == date:
                        found = True
                        current_number = int(sheet.cell(row_index + 1, number_column).value)
                        new_number = current_number + 1
                        sheet.update_cell(row_index + 1, number_column, str(new_number))
                        break

                if found == False and new_issue != "greeting" and new_issue != "goodbye":
                    # Find the next empty row
                    next_row = len(issue_values) + 1
                    for row in range(1, sheet.row_count + 1):
                        if not sheet.cell(row, issue_column).value:
                            next_row = row
                            break

                    sheet.update_cell(next_row, issue_column, new_issue)
                    sheet.update_cell(next_row, number_column, "1")
                    sheet.update_cell(next_row, date_column, date)
                    issue_values.append(new_issue)
                return random.choice(data["responses"])
    else:
        return "Please visit us for assistance."

if __name__ == "__main__":
    print("Let's chat! type 'quit' to exit")
    while True:
        sentence = input('You: ')
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
