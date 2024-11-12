import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(self, a, b, k, stride=1, device=None, dtype="float32"):
        super().__init__()
        self.conv = nn.Conv(a, b, k, stride=stride, device=device, dtype=dtype)
        self.bn = nn.BatchNorm2d(b, device=device, dtype=dtype)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()

        self.conv1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype)
        self.conv2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype)

        self.conv3 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        self.conv4 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)

        self.conv5 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype)
        self.conv6 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype)

        self.conv7 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)
        self.conv8 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype)

        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(128, 128, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(128, 10, device=device, dtype=dtype)


    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)

        v3 = self.conv3(v2)
        v4 = self.conv4(v3)

        v4 = v4 + v2

        v5 = self.conv5(v4)
        v6 = self.conv6(v5)

        v7 = self.conv7(v6)
        v8 = self.conv8(v7)

        v8 = v8 + v6
        v8 = self.flatten(v8)
        
        v9 = self.lin1(v8)
        v10 = self.relu(v9)
        v11 = self.lin2(v10)

        return v11


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
