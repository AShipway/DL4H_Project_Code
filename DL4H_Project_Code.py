import torch
import tqdm as notebook_tqdm
import json

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gensim
from gensim.models import Word2Vec

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


#Load training and test json output of CoreNLP
with open("MSRP_train15p.json", "r") as read_file:
    MSRP_15p = json.load(read_file)
with open("MSRP_test5p.json", "r") as read_file:
    MSRP_5p = json.load(read_file)

core_words = []
max_sen_length = 0

core_words_test = []
max_sen_length_test = 0

#Create SOP kernels from nouns and verbs
for i in range(len(MSRP_15p['sentences'])):
    temp_list = []
    for j in MSRP_15p['sentences'][i]['tokens']:
        if 'NN' in j['pos'] or 'VB' in j['pos']:
            temp_list.append(j['originalText'].lower())
    core_words.append(temp_list)
    if len(temp_list) > max_sen_length:
        max_sen_length = len(temp_list)

for i in range(len(MSRP_5p['sentences'])):
    temp_list = []
    for j in MSRP_15p['sentences'][i]['tokens']:
        if 'NN' in j['pos'] or 'VB' in j['pos']:
            temp_list.append(j['originalText'].lower())
    core_words_test.append(temp_list)
    if len(temp_list) > max_sen_length_test:
        max_sen_length_test = len(temp_list)

#Separate sentence pairs into separate lists
core_words1 = []
core_words2 = []
for i in range(len(core_words)):
    if i%2 == 0:
        continue
    core_words1.append(core_words[i])
for i in range(len(core_words)):
    if i == 0 or i%2 == 0:
        core_words2.append(core_words[i])

core_words1_test = []
core_words2_test = []
for i in range(len(core_words_test)):
    if i%2 == 0:
        continue
    core_words1_test.append(core_words_test[i])
for i in range(len(core_words_test)):
    if i == 0 or i%2 == 0:
        core_words2_test.append(core_words_test[i])

#Create Word2vec bag of words model with training set words

wv_model = gensim.models.Word2Vec(core_words, vector_size=50, workers=4, min_count=1)

#Create sentence tensor arrays

sentence_tensors = []
sentence_tensors1 = []
sentence_tensors2 = []
sentence_masks = []
sentence_masks1 = []
sentence_masks2 = []
zero_mask = torch.zeros(50)
zero_mask = torch.reshape(zero_mask, (-1,1))
ones_mask = torch.ones(50)
ones_mask = torch.reshape(ones_mask, (-1,1))

sentence_tensors1_test = []
sentence_tensors2_test = []
sentence_masks1_test = []
sentence_masks2_test = []

for sen in core_words:
    temp_list = []
    temp_mask = []
    for word in sen:
        w_vec = torch.tensor(wv_model.wv[word])
        w_vec = torch.reshape(w_vec, (-1,1))
        temp_list.append(w_vec)
        temp_mask.append(ones_mask)
    if len(temp_list) < max_sen_length:
        for i in range(len(temp_list), max_sen_length):
            temp_list.append(zero_mask)
            temp_mask.append(zero_mask)
    temp_stack = torch.stack(temp_list, dim = 0)
    temp_stack = torch.squeeze(temp_stack)
    temp_stack = torch.unsqueeze(temp_stack, 0)
    mask_stack = torch.stack(temp_mask, dim = 0)
    mask_stack = torch.squeeze(mask_stack)
    mask_stack = torch.unsqueeze(mask_stack, 0)
    sentence_tensors.append(temp_stack)
    sentence_masks.append(mask_stack)

sentences = torch.stack(sentence_tensors)
print(sentences.shape)
masks = torch.stack(sentence_masks)


for sen in core_words1:
    temp_list = []
    temp_mask = []
    for word in sen:
        w_vec = torch.tensor(wv_model.wv[word])
        w_vec = torch.reshape(w_vec, (-1,1))
        temp_list.append(w_vec)
        temp_mask.append(ones_mask)
    if len(temp_list) < max_sen_length:
        for i in range(len(temp_list), max_sen_length):
            temp_list.append(zero_mask)
            temp_mask.append(zero_mask)
    temp_stack = torch.stack(temp_list, dim = 0)
    temp_stack = torch.squeeze(temp_stack)
    temp_stack = torch.unsqueeze(temp_stack, 0)
    mask_stack = torch.stack(temp_mask, dim = 0)
    mask_stack = torch.squeeze(mask_stack)
    mask_stack = torch.unsqueeze(mask_stack, 0)
    sentence_tensors1.append(temp_stack)
    sentence_masks1.append(mask_stack)


sentences1 = torch.stack(sentence_tensors1)
print(sentences.shape)
masks1 = torch.stack(sentence_masks1)


for sen in core_words2:
    temp_list = []
    temp_mask = []
    for word in sen:
        w_vec = torch.tensor(wv_model.wv[word])
        w_vec = torch.reshape(w_vec, (-1,1))
        temp_list.append(w_vec)
        temp_mask.append(ones_mask)
    if len(temp_list) < max_sen_length:
        for i in range(len(temp_list), max_sen_length):
            temp_list.append(zero_mask)
            temp_mask.append(zero_mask)
    temp_stack = torch.stack(temp_list, dim = 0)
    temp_stack = torch.squeeze(temp_stack)
    temp_stack = torch.unsqueeze(temp_stack, 0)
    mask_stack = torch.stack(temp_mask, dim = 0)
    mask_stack = torch.squeeze(mask_stack)
    mask_stack = torch.unsqueeze(mask_stack, 0)
    sentence_tensors2.append(temp_stack)
    sentence_masks2.append(mask_stack)


sentences2 = torch.stack(sentence_tensors2)
print(sentences.shape)
masks2 = torch.stack(sentence_masks2)

for sen in core_words1_test:
    temp_list = []
    temp_mask = []
    for word in sen:
        w_vec = torch.tensor(wv_model.wv[word])
        w_vec = torch.reshape(w_vec, (-1,1))
        temp_list.append(w_vec)
        temp_mask.append(ones_mask)
    if len(temp_list) < max_sen_length:
        for i in range(len(temp_list), max_sen_length):
            temp_list.append(zero_mask)
            temp_mask.append(zero_mask)
    temp_stack = torch.stack(temp_list, dim = 0)
    temp_stack = torch.squeeze(temp_stack)
    temp_stack = torch.unsqueeze(temp_stack, 0)
    mask_stack = torch.stack(temp_mask, dim = 0)
    mask_stack = torch.squeeze(mask_stack)
    mask_stack = torch.unsqueeze(mask_stack, 0)
    sentence_tensors1_test.append(temp_stack)
    sentence_masks1_test.append(mask_stack)

sentences1_test = torch.stack(sentence_tensors1_test)
print(sentences1_test.shape)
masks1_test = torch.stack(sentence_masks1_test)


for sen in core_words2_test:
    temp_list = []
    temp_mask = []
    for word in sen:
        w_vec = torch.tensor(wv_model.wv[word])
        w_vec = torch.reshape(w_vec, (-1,1))
        temp_list.append(w_vec)
        temp_mask.append(ones_mask)
    if len(temp_list) < max_sen_length:
        for i in range(len(temp_list), max_sen_length):
            temp_list.append(zero_mask)
            temp_mask.append(zero_mask)
    temp_stack = torch.stack(temp_list, dim = 0)
    temp_stack = torch.squeeze(temp_stack)
    temp_stack = torch.unsqueeze(temp_stack, 0)
    mask_stack = torch.stack(temp_mask, dim = 0)
    mask_stack = torch.squeeze(mask_stack)
    mask_stack = torch.unsqueeze(mask_stack, 0)
    sentence_tensors2_test.append(temp_stack)
    sentence_masks2_test.append(mask_stack)


sentences2_test = torch.stack(sentence_tensors2_test)
print(sentences1_test.shape)
masks2_test = torch.stack(sentence_masks2_test)


#Create target ground truth tensor
target_temp2 = [0,1,0,1,0,0,1,0,1,0,1,1,1,0,0]
target2 = torch.tensor(target_temp2).float()

target_test_temp = [1,1,0,1,0]
target_test = torch.tensor(target_test_temp).float()

#K-max pooling method

def kmax_pooling(x, dim, k):
    index = torch.topk(x, k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index)


#CNN model

class CNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # your code here
        
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.conv2 = nn.Conv2d(5, 5, 3, padding = 1)
        self.pool = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(720, 50)
        self.pwd = nn.PairwiseDistance(p=1)
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-08)
        
        #raise NotImplementedError

    def forward(self, x1, x2):
        #input is of shape (batch_size=32, 3, 224, 224) if you did the dataloader right
        # your code here
        
        x1 = F.relu(self.conv1(x1))
        x1 = kmax_pooling(x1, 2, 3)
        #x1 = self.pool(x1)
        x1 = F.relu(self.conv2(x1))
        
        x1 = kmax_pooling(x1, 2, 3)
        x1 = torch.flatten(x1, 1) # flatten all dimensions except batch
        x1 = self.fc1(x1)
        
        x2 = F.relu(self.conv1(x2))
        x2 = kmax_pooling(x2, 2, 3)
        #x2 = self.pool(x2)
        x2 = F.relu(self.conv2(x2))
        x2 = kmax_pooling(x2, 2, 3)
        x2 = torch.flatten(x2, 1) # flatten all dimensions except batch
        x2 = self.fc1(x2)
        
        result = self.pwd(x1, x2)

        return result


model = CNN()

criterion = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr = np.exp(-1))


#Model Training

n_epochs = 30

def train_model(model, train_data1, train_data2, target, n_epoch=n_epochs, optimizer=optimizer, criterion=criterion):
    import torch.optim as optim
    """
    :param model: A CNN model
    :param train_dataloader: the DataLoader of the training data
    :param n_epoch: number of epochs to train
    :return:
        model: trained model
    """
    model.train() # prep model for training
    
    
    for epoch in range(n_epoch):
        curr_epoch_loss = []
        
        y_hat = model(train_data1, train_data2)
        
        print(y_hat)
        
        y_hat = torch.reshape(y_hat, (-1,1))

        target = torch.reshape(target, (-1,1))
        
        loss = criterion(y_hat, target)
            
        optimizer.zero_grad()
            
        loss.backward()
            
        optimizer.step()
        
        curr_epoch_loss.append(loss.cpu().data.numpy())
        
        print(f"Epoch {epoch}: curr_epoch_loss={np.mean(curr_epoch_loss)}")

    return model


seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model = train_model(model, sentences1, sentences2, target2)


#Testing

#Testing
def eval_model(model, test_data1, test_data2, target_test):
    """
    :return:
        Y_pred: prediction of model on the dataloder.
            Should be an 2D numpy float array where the second dimension has length 2.
        Y_test: truth labels. Should be an numpy array of ints
    TODO:
        evaluate the model using on the data in the dataloder.
        Add all the prediction and truth to the corresponding list
        Convert Y_pred and Y_test to numpy arrays (of shape (n_data_points, 2))
    """
    model.eval()
    Y_pred = []
    Y_test = []

        # your code here
        
    y_hat = model(test_data1, test_data2)
        
    y_pred = (y_hat > 0.5).type(torch.float)
    
    Y_pred = torch.reshape(y_pred, (-1,1))

    Y_test = torch.reshape(target_test, (-1,1))
        

    return Y_pred, Y_test

#Evaluation
y_pred, y_true = eval_model(model, sentences1_test, sentences2_test, target_test)
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
print(("Validation Accuracy: " + str(acc)))
print(("Validation F1: " + str(f1)))


        



