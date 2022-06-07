# Copyright (c) 2017-2019 Keio University.
# Authors: Mineto Tsukada and Hiroki Matsutani
# Keio Ref. No. CR-0081

import numpy as np
import pickle

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Network definition


class OS_ELM(object):
    def __init__(self, inputs, units, outputs, activation='sigmoid', loss='mean_squared_error'):
        self.inputs = inputs
        self.units = units
        self.outputs = outputs
        # Modified by Matsutani (2019-08-10)
        sigmoid_range = 34.538776394910684  # np.log((1 - 1e-15) / 1e-15)
        scale = sigmoid_range / inputs * 2.0
        self.alpha = np.random.rand(
            inputs, units) * scale * 2 - scale  # [-scale, +scale]
        self.beta = np.random.rand(units, outputs) * \
            scale * 2 - scale  # [-scale, +scale]
        #
        self.bias = np.zeros(shape=(1, self.units))
        self.p = None
        # Added by Matsutani (2018-08-15)
        self.forget = 0.5
        #
        if loss == 'mean_squared_error':
            self.lossfun = self.__mean_squared_error
        elif loss == 'l1_error':
            self.lossfun = self.__l1_error
        else:
            raise Exception('unknown loss function was specified.')
        if activation == 'sigmoid':
            self.actfun = self.__sigmoid
        elif activation == 'relu':
            self.actfun = self.__relu
        elif activation == 'linear':
            self.actfun = self.__identify
        else:
            raise Exception('unknown activation function was specified.')

    def __mean_squared_error(self, out, y):
        return 0.5 * np.mean((out - y)**2)

    def __l1_error(self, out, y):
        return np.mean(np.abs((out - y)))

    def __accuracy(self, out, y):
        batch_size = len(out)
        accuracy = np.sum((np.argmax(out, axis=1) == np.argmax(y, axis=1)))
        return accuracy / batch_size

    def __identify(self, x):
        return x

    def __sigmoid(self, x):
        # Added by Matsutani (2019-08-06)
        # Restrict domain of sigmoid function within [1e-15, 1 - 1e-15]
        sigmoid_range = 34.538776394910684  # np.log((1 - 1e-15) / 1e-15)
        x = np.clip(x, -sigmoid_range, sigmoid_range)
        #
        return 1.0 / (1.0 + np.exp(-x))

    def __relu(self, x):
        return np.maximum(0, x)

    def __softmax(self, x):
        c = np.max(x, axis=1).reshape(-1, 1)
        upper = np.exp(x - c)
        lower = np.sum(upper, axis=1).reshape(-1, 1)
        return upper / lower

   
    def __call__(self, x):
        #print(self.alpha.shape,end=" ")
        h1 = x.dot(self.alpha) + self.bias
        a1 = self.actfun(h1)
        out = a1.dot(self.beta)
        return out

    def forward(self, x):
        return self(x)

    def classify(self, x):
        return self.__softmax(self(x))

    def compute_accuracy(self, x, y):
        out = self(x)
        acc = self.__accuracy(out, y)
        return acc

    def compute_loss(self, x, y):
        out = self(x)
        loss = self.lossfun(out, y)
        return loss

    def init_train(self, x, y):
        """assert len(x) >= self.units, 'initial dataset size must be >= %d' % (
            self.units)"""
        H = self.actfun(x.dot(self.alpha) + self.bias)
        HT = H.T
        self.p = np.linalg.pinv(HT.dot(H))
        self.beta = self.p.dot(HT).dot(y)

    def seq_train(self, x, y):

        H = self.actfun(x.dot(self.alpha))
        # np.set_printoptions(suppress=True)

        HT = H.T
        I = np.eye(len(x))  # I.shape = (N, N) N:length of inputa data

        # Added by Matsutani (2018-08-15)
        self.p = self.p / (self.forget * self.forget)
        #
        # Update P
        temp = np.linalg.pinv(I + H.dot(self.p).dot(HT))  # temp.shape = (N, N)

        self.p = self.p - (self.p.dot(HT).dot(temp).dot(H).dot(self.p))
        # Update beta

        self.beta = self.beta + (self.p.dot(HT).dot(y - H.dot(self.beta)))

    def save_weights(self, path):
        weights = {
            'alpha': self.alpha,
            'beta': self.beta,
            'p': self.p}
        with open(path, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, path):
        with open(path, 'rb') as f:
            weights = pickle.load(f)
            self.alpha = weights['alpha']
            self.beta = weights['beta']
            self.p = weights['p']

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    # Added by Matsutani (2018-08-15)
    def set_forget(self, forget):
        self.forget = forget
    #

    ######
    # added watanabe

    def compute_softmax(self, out):
        print("out ", out)
        max_out = np.max(out)
        print("max_out ", max_out)
        exp_out = np.exp(out - max_out)
        print("exp_out ", exp_out)
        sum_exp_lower = np.sum(exp_out)
        return exp_out / sum_exp_lower

    def get_pi(self, s):
        out = self(s)
        return self.compute_softmax(out)

    def get_action(self, s):
        pi = self.get_pi(s)
        action = np.random.choice([0, 1], p=pi[0])
        return action

    def get_log_prob(self, s, a):
        pi = self.get_pi(s)
        return np.log(pi + 1e-15)[0][a]

    def set_p(self):
        """H = self.actfun(x.dot(self.alpha) + self.bias)
        HT = H.T
        self.p = np.linalg.pinv(HT.dot(H))"""
        self.p = np.random.uniform(0, 1, (self.units, self.units)) * 0.0001





