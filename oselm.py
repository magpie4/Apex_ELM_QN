import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class OS_ELM(object):
    def __init__(self, inputs, units, outputs, weight_decay=1.0, lipchitz_alpha_flag=True):
        self.inputs = inputs
        self.units = units
        self.outputs = outputs
        self.steps = 0
        self.weight_decay = weight_decay
        self.lipchitz_alpha_flag = lipchitz_alpha_flag

        sigmoid_range = 34.538776394910684
        self.scale = sigmoid_range / inputs * 2.0

        self.alpha = np.random.uniform(
            0, 1, (inputs, units)).astype(np.float32)
        if lipchitz_alpha_flag is True:
            self.alpha = self.lipchitz_norm_alpha(self.alpha)
        self.beta = np.random.uniform(
            0, 1, (units, outputs)).astype(np.float32)
        self.bias = np.zeros(shape=(1, units)).astype(np.float32)
        self.p = None
        self.forget = 1

    def forward(self, x):
        return self(x)

    def sigmoid(self, x):
        ret_value = 1 / (1 + np.exp(x))
        return ret_value

    def __call__(self, x):
        h1 = x @ self.alpha + self.bias
        a1 = self.relu(h1)
        out = a1 @ self.beta
        return out

    def seq_train_with_forget(self, x, y, forget_rate=0.99):
        H = self.relu(x @ self.alpha)
        H = H + np.eye(len(x)) + 1e-8
        HT = H.T
        I = np.eye(len(x))
        self.p = (1 / forget_rate) * self.p - \
            (1 / forget_rate) * ((self.p @ HT @ H @ self.p) /
                                 (forget_rate + H @ self.p @ HT))
        self.beta = self.beta + self.p @ HT @ (y - H @ self.beta)

    def seq_train(self, x, y, t):
        H = self.relu(x.dot(self.alpha))
        H = H + np.eye(H.shape[0]) * 1e-8
        HT = H.T
        I = np.eye(len(x))  # I.shape = (N, N) N:length of inputa data
        # Added by Matsutani (2018-08-15)
        self.p = self.p / (self.forget * self.forget)
        #
        # Update P
        temp = np.linalg.pinv(I + H.dot(self.p).dot(HT))  # temp.shape = (N, N)
        self.p = self.p - (self.p.dot(HT).dot(temp).dot(H).dot(self.p))
        print()
        # Update beta
        self.beta = self.beta + (self.p.dot(HT).dot(y - H.dot(self.beta)))
        

    def init_train(self, x, y):
  
        H = self.relu(x.dot(self.alpha) + self.bias)
        H_alt = H = self.relu(x @ self.alpha + self.bias)
        H = H + np.eye(len(x)) * 1e-8
        HT = H.T

        self.p = np.linalg.pinv(
            HT.dot(H) + np.eye(H.shape[0]) * self.weight_decay)

        self.beta = self.p @ HT @ y


    def lipchitz_norm_alpha(self, x):
        _, singular_values, _ = np.linalg.svd(x, full_matrices=False)
        max_singular_value = singular_values[0]
        return x / max_singular_value


    def set_p(self):
        self.p = np.random.uniform(0, 1, (self.units, self.units)) * 0.0001

    def relu(self, x):
        x[x < 0] = 0
        return x
    
    def set_weight(self, weight):
        self.alpha = weight[0]
        self.beta = weight[1]
    
    def get_weight(self):
        return [self.alpha,self.beta]
        
