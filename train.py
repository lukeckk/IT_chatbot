import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet

with open('dataset.json', 'r') as file:
    dataset = json.load(file)

all_words = []
tags = []   # store different tags
xy = []     #store the patterns

for data in dataset['intents']:
    tag = data['tag']
    tags.append(tag)
    for pattern in data['patterns']:
        word = tokenize(pattern)    # tokenize the questions
        all_words.extend(word)   # add the content of list another list to prevent 2d array
        xy.append((word, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(word) for word in all_words if word not in ignore_words]  # exclude special characters
all_words = sorted(set(all_words)) # sort and remove duplicates
tags = sorted(set(tags)) # remove duplicate tags, optional
# print(all_words)
# print(tags)

# creating bag of words
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:   #looping through xy list
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag) # get the index of each tag 0 = delivery, 1 = funny, 2 = goodbye, etc...
    y_train.append(label) # CrossEntropyLoss
    # print(tag)

x_train = np.array(x_train)   # covert array to array object without comma for NLP
y_train = np.array(y_train)   # covert array to array object without comma for NLP

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
input_size = len(x_train[0]) # length of bag_of_words
hidden_size = 8
output_size = len(tags) # number of classes
# print(input_size, len(all_words))
# print(output_size, tags)
learning_rate = 0.001
num_epochs = 1000

chat = ChatDataset()
train_loader = DataLoader(dataset=chat, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # tell pytorch to use cuda is available, else use cpu
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# model training loop
for epoch in range (num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

# save the data to pytorch file to be open in chat.py
data = {
     "model_state": model.state_dict(),
     "input_size": input_size,
     "output_size": output_size,
     "hidden_size": hidden_size,
     "all_words": all_words,
     "tags": tags
 }
FILE = "data.pth"
torch.save(data, FILE)
print(f'training complete. file saved to {FILE}')