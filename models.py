# models.py

from tkinter import X
from typing import Any
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class FFNN(nn.Module):
    def __init__(self, inp, hid, out, embed: WordEmbeddings):
        super(FFNN, self).__init__()
        self.embeddings = embed
        self.embedLayer = embed.get_initialized_embedding_layer()

        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, hid)
        self.g1 = nn.ReLU()
        self.W1 = nn.Linear(hid, hid)
        self.g2 = nn.ReLU()
        self.W2 = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        

    def forward(self, sentence, printThis=False):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """

        
        # DAN

        input=[]
        for word in sentence:
            index = self.embeddings.word_indexer.index_of(word)
            
            if index == -1:
                input.append(0)
            else:
                input.append(index)
            # input += 
        # print(len(input))
        z = self.embedLayer(torch.LongTensor(input))
        # print(z.size())
        z = torch.mean(z, 0)
        # if(printThis): print(z)
        z = self.V(z)
        z = self.g(z)
        z = self.W(z)
        # if(printThis): print("after hidden layer",z)
        # z = self.g1(z)
        # z = self.W1(z)
        z = self.g2(z)
        z = self.W2(z)
        if(printThis): print("before softmax",z)
        return self.log_softmax(z)
    
    # def back(self, output, target):
    #     loss = nn.NLLLoss()
    #     # print("output",output)
    #     # print("target",target)
    #     x = loss(output, target)
    #     # print("loss",x)
    #     x.backward()
    #     self.optimizer.step()


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    def __init__(self, embed: WordEmbeddings):
        self.embed = embed
        self.NN = FFNN(embed.get_embedding_length(), 50, 1, embed)

    def getprob(self, ex_words: List[str]) -> any:
        result = self.NN.forward(ex_words)
        return result

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        # print(len(ex_words))
        answer = self.NN.forward(ex_words, True)
        # print(answer[0])
        
        # if answer[0]>answer[1]:
        #     return 0
        # return 1


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    lossFunction = nn.NLLLoss()
    classifier = NeuralSentimentClassifier(word_embeddings)
    optimizer = optim.Adam(classifier.NN.parameters(), lr=.1)
    random.shuffle(train_exs)
    train_exs = train_exs[:1000]
    print(classifier.NN.parameters())
    for epoch in range(3):
        totalLoss = 0
        random.shuffle(train_exs)
        for ex in train_exs:
            words, target = ex.words, ex.label
            prob = classifier.getprob(words)
            
            # correct: Any
            # if ex.label == 0:
            #     # print(0)
            #     correct = torch.tensor([1, 0])
            # else:
            #     # print(1)
            #     correct = torch.tensor([0, 1])
            # # print(prob)
            x = torch.LongTensor(target)
            # print(x.size())
            # print(prob.size())

            loss = lossFunction(prob,x)
            # print(loss)
            totalLoss+=loss
            classifier.NN.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"loss in {epoch} is {totalLoss}")
        print(classifier.predict(train_exs[0].words))
        print(classifier.predict(train_exs[1].words))
    #     print(prob)
    # print(classifier.predict(train_exs[0].words))

    return classifier
