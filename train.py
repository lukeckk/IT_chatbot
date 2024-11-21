import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet

with open('dataset.json', 'r') as file:
    dataset = json.load(file)

all_words = [] # store each patterns word as token
tags = []   # store different tags
xy = []     # tokenized word as key, tag as value

# tokenizing
for data in dataset['intents']:
    tag = data['tag']
    tags.append(tag)
    for pattern in data['patterns']:
        word = tokenize(pattern)    # tokenize the questions
        all_words.extend(word)   # add the content of list another list to prevent 2d array
        xy.append((word, tag))

# stemming word and removing symbol and duplicates
ignore_words = ['?', '!', '.', ',']
all_words = [stem(word) for word in all_words if word not in ignore_words]  # exclude special characters
all_words = sorted(set(all_words)) # sort and remove duplicates
tags = sorted(set(tags)) # remove duplicate tags, optional

# creating bag of words
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:   #looping through xy list
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    label = tags.index(tag) # get the index of each tag 0 = delivery, 1 = funny, 2 = goodbye, etc...
    y_train.append(label) # CrossEntropyLoss

# covert array to array object without comma for NLP
x_train = np.array(x_train)
y_train = np.array(y_train)

# boiler code to be used for DataLoader
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
batch_size = 8 # number of samples processed together each time
hidden_size = 8 # number of neuron in hidden layers
input_size = len(x_train[0]) # length of bag_of_words
output_size = len(tags) # number of tags
learning_rate = 0.001 # step size for updating parameters during training. 0.001 is typical for Adam
num_epochs = 1000

chat = ChatDataset()
train_loader = DataLoader(dataset=chat, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # tell pytorch to use cuda is available, else use cpu
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss function for classification
criterion = nn.CrossEntropyLoss()
# optimizer 'Adam' for adjust model parameters based on gradients computed during backpropagation
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# model training loop of 1000
for epoch in range (num_epochs):
    for(words, labels) in train_loader:
        words = words.to(device) # bag of words
        labels = labels.to(device) # tags

        # forward pass to compute predictions
        outputs = model(words)

        # find the difference between output and labels using loss function
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad()

        # PyTorch applies the chain rule of calculus to compute gradients for every parameter.
        loss.backward()

        # update parameter
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