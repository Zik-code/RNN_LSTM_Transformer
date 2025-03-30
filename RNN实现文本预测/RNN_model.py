# RNN_model.py
import torch
from torch import nn
 
 
class Model(nn.Module):
    """
    input_size (int):输入数据的特征大小，即每个时间步的输入向量的维度。
    hidden_size (int):隐藏层的特征大小，即每个时间步的隐藏状态向量的维度。
    num_layers (int,可选):RNN的层数，默认值为1。当层数大于1时，RNN会变为多层RNN。
    nonlinearity (str,可选):指定激活函数，默认值为'tanh'。可选值有'tanh'和'relu'。
    bias (bool,可选):如果设置为True，则在RNN中添加偏置项。默认值为True。
    batch_first (bool,可选):如果设置为True，则输入数据的形状为(batch_size, seq_len, input_size)。否则，默认输入数据的形状为(seq_len, batch_size, input_size)。默认值为False。
    dropout (float,可选):如果非零，则在除最后一层之外的每个RNN层之间添加dropout层，其丢弃概率为dropout。默认值为0。
    bidirectional (bool,可选):如果设置为True，则使用双向RNN。默认值为False。
    """
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()
 
        # Defining some parameters
        self.hidden_dim = hidden_dim # 隐藏状态 ht 的维度
        self.n_layers = n_layers # 网络的层数
 
        # Defining the layers
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
 
    def forward(self, x):
        batch_size = x.size(0)
 
        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)
 
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x, hidden)
 
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
 
        return out, hidden
 
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return hidden
 