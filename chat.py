import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

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


# name the bot
bot_name = "IT Helper"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenize(sentence)   #tokenize user's input
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
                print(f'{bot_name}: {random.choice(data["responses"])}')     # generate random response from each tag
    else:
        print(f'{bot_name}: I do not understand...')
