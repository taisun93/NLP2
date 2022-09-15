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
        self.embed = embed
        

        self.V = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.g = nn.ReLU()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.xavier_uniform_(self.V.weight)
        nn.init.xavier_uniform_(self.W.weight)
        self.optimizer = optim.Adam(self.parameters(), lr=0.1)

    def forward(self, sentence):
        """
        Runs the neural network on the given data and returns log probabilities of the various classes.

        :param x: a [inp]-sized tensor of input data
        :return: an [out]-sized tensor of log probabilities. (In general your network can be set up to return either log
        probabilities or a tuple of (loss, log probability) if you want to pass in y to this function as well
        """

        self.zero_grad()
        x = self.embed.get_embedding(sentence[0])

        for word in sentence[1:]:
            x += self.embed.get_embedding(word)

        x /= len(sentence)
        b = torch.from_numpy(x).float()
        return self.log_softmax(self.W(self.g(self.V(b))))
    
    def back(self, output, target):
        loss = nn.NLLLoss()
        print("output",output)
        print("target",target)
        x = loss(output, target)
        print("loss",x)
        x.backward()
        self.optimizer.step()


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

    def __init__(self, embed: WordEmbeddings):
        self.embed = embed
        self.NN = FFNN(embed.get_embedding_length(), 50, 2, embed)

    def getprob(self, ex_words: List[str]) -> any:
        result = self.NN.forward(ex_words)
        return result

    def correct(self, result, target):
        # loss = torch.neg(result).dot(correct)
        self.NN.back(result, target)
        # return



    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        # print(len(ex_words))
        answer = self.getprob(ex_words).detach().numpy()
        # print(answer[0])
        # # print(answer[])
        # print(answer)
        if answer[0]>answer[1]:
            return 0
        # print("sdfljksdfjkldsfjlk")
        return 1


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample], word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """

    # wordembeder = read_word_embeddings(args.word_vecs_path)
    # print(word_embeddings.get_initialized_embedding_layer().embedding_dim)
    classifier = NeuralSentimentClassifier(word_embeddings)
    # zoop = train_exs[0].words[0]
    # word_embeddings.get_embedding(zoop)
    # torch.from_numpy(train_exs[0]).float()
    for epoch in range(5):
        random.shuffle(train_exs)
        for ex in train_exs:
            prob = classifier.getprob(ex.words)
            right: Any
            if ex.label == 0:
                right = torch.tensor([1, 0])
            else:
                right = torch.tensor([0, 1])

            classifier.correct(prob,right)
        
    # for ex in train_exs[:2]:
    #     prob = classifier.getprob(ex.words)
    #     correct: Any
    #     if ex.label == 0:
    #         correct = torch.tensor([1.0, 0.0])
    #     else:
    #         correct = torch.tensor([0.0, 1.0])

    #     classifier.correct(prob,correct)
    #     print(prob)
    classifier.predict(train_exs[0].words)

    return classifier
