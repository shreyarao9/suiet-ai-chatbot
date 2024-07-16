import json
from nltk__utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
import random

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
#loop through each sentence in intents patterns
for intent in intents['intents']:#since the json file has a key 'intents' under which all the data is
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w) #not using append because it's an array and not an array of arrays
        xy.append((w, tag)) #tuple

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) #to remove duplicates
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag) #so X has bag of words for each pattern sentence

    label = tags.index(tag) #numbers for the labels / tags
    y_train.append(label) #PyTorch CrossEntropyLoss needs only class labels, not one-hot (?)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):#inherits Dataset
    def __init__(self): #implements init function, which only gets self
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
#hyperparameters
num_epochs = 1000
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0]) # len of all_words could work here too
learning_rate = 0.001
print(input_size, output_size)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cpu') # since cuda isn't available on my laptop. You can use torch.cuda.is_available() and pass 'cuda' if True

model = NeuralNet(input_size, hidden_size, output_size).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')