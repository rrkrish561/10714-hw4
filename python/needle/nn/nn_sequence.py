"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return ops.power_scalar(1 + ops.exp(-x), -1)

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        a = 1 / np.sqrt(hidden_size)

        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-a, high=a, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-a, high=a, device=device, dtype=dtype))

        self.nonlinearity = nonlinearity

        if bias:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-a, high=a, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-a, high=a, device=device, dtype=dtype))

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        bs = X.shape[0]
        hidden_size = self.W_ih.shape[1]

        acc = X @ self.W_ih
        if h is not None:
            acc += h @ self.W_hh

        if hasattr(self, 'bias_ih'):
            acc += ops.broadcast_to(ops.reshape(self.bias_ih + self.bias_hh, (1, hidden_size)), (bs, hidden_size))

        if self.nonlinearity == 'tanh':
            h = ops.tanh(acc)
        elif self.nonlinearity == 'relu':
            h = ops.relu(acc)

        return h


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
            else:
                self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device))

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        seq_len, bs, input_size = X.shape

        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
        h = h0

        hidden = ops.split(h, 0)
        out_tuple = ops.split(X, 0)

        out_hidden = []

        for i in range(self.num_layers):
            new_out_tuple = []
            new_h = hidden[i]
            for j in range(seq_len):
                new_h = self.rnn_cells[i](out_tuple[j], new_h)
                new_out_tuple.append(new_h)
            
            out_hidden.append(new_h)
            out_tuple = tuple(new_out_tuple)

        return ops.stack(out_tuple, 0), ops.stack(tuple(out_hidden), 0)


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.sigmoid = Sigmoid()

        a = 1 / np.sqrt(hidden_size)

        self.W_ih = Parameter(init.rand(input_size, 4*hidden_size, low=-a, high=a, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4*hidden_size, low=-a, high=a, device=device, dtype=dtype))
        
        if bias:
            self.bias_ih = Parameter(init.rand(4*hidden_size, low=-a, high=a, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(4*hidden_size, low=-a, high=a, device=device, dtype=dtype))


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        acc = X @ self.W_ih

        if h is not None:
            acc += h[0] @ self.W_hh
        
        if hasattr(self, 'bias_ih'):
            acc += ops.broadcast_to(ops.reshape(self.bias_ih + self.bias_hh, (1, 4*self.hidden_size)), (X.shape[0], 4*self.hidden_size))

        acc = ops.reshape(acc, (X.shape[0], 4, self.hidden_size))
        gates = ops.split(acc, 1)

        i = self.sigmoid(gates[0])
        f = self.sigmoid(gates[1])
        g = ops.tanh(gates[2])
        o = self.sigmoid(gates[3])

        c = i * g
        if h is not None:
            c += f * h[1]

        h = o * ops.tanh(c)

        return h, c


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm_cells = []
        for i in range(num_layers):
            if i == 0:
                self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
            else:
                self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        
        seq_len, bs, input_size = X.shape

        if h is None:
            h = init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype), init.zeros(self.num_layers, bs, self.hidden_size, device=X.device, dtype=X.dtype)
        
        hidden, cell = h
        hidden = ops.split(hidden, 0)
        cell = ops.split(cell, 0)

        out_hidden = []
        out_cell = []

        out_tuple = ops.split(X, 0)

        for i in range(self.num_layers):
            new_out_tuple = []
            new_h = hidden[i]
            new_c = cell[i]
            for j in range(seq_len):
                new_h, new_c = self.lstm_cells[i](out_tuple[j], (new_h, new_c))
                new_out_tuple.append(new_h)
            
            out_hidden.append(new_h)
            out_cell.append(new_c)
            out_tuple = tuple(new_out_tuple)

        output = ops.stack(out_tuple, 0)

        return output, (ops.stack(tuple(out_hidden), 0), ops.stack(tuple(out_cell), 0))

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION