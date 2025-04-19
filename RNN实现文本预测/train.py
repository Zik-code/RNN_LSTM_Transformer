 
# train.py
import torch
from torch import nn
 
import numpy as np
 
# 首先，我们将定义我们希望模型在输入第一个单词或前几个字符时输出的句子。
# 然后我们将从句子中的所有字符创建一个字典，并将它们映射到一个整数。
# 这将允许我们将输入字符转换为它们各自的整数（char2int），反之亦然（int2char）。
 
text = ['hey how are you', 'good i am fine', 'have a nice day']
 
chars = set(''.join(text))
# print(chars)# 输出 : {'y', 'o', ' ', 'd', 'f', 'n', 'm', 'i', 'w', 'r', 'u', 'v', 'h', 'c', 'g', 'e', 'a'} (注意:输出不定，但都包含了所有的字符)

int2char = dict(enumerate(chars))
# print('int2char=',int2char)
# 输出：int2char= {0: 'a', 1: 'r', 2: 'u', ············}
 
# 对 int2char反转
char2int = {char: ind for ind, char in int2char.items()}
# print(char2int)
# 输出：char2int = {'e': 0, 'y': 1, 'm': 2, ···········}
 
# ------------------------------------------------------------------------------------
# 接下来，我们将填充(padding)输入句子以确保所有句子都是标准长度。
# 虽然 RNN 通常能够接收可变大小的输入，但我们通常希望分批输入训练数据以加快训练过程。
# 为了使用批次(batch)来训练我们的数据，我们需要确保输入数据中的每个序列大小相等。
 
# 因此，在大多数情况下，可以通过用 0 值填充太短的序列和修剪太长的序列来完成填充。
# 在我们的例子中，我们将找到最长序列的长度，并用空格填充其余句子以匹配该长度。
 
# Finding the length of the longest string in our data
maxlen = len(max(text, key=len))
print('maxlen=',maxlen)
# 输出 maxlen= 15，找到最长的句子hey how are 占15个字符。

# 填充长度不足15的句子为15
for i in range(len(text)):
  while len(text[i])<maxlen:
      text[i] += ' '
 
# 由于我们要在每个时间步预测序列中的下一个字符，我们必须将每个句子分为：
 
# 输入数据
# 最后一个字符需排除因为它不需要作为模型的输入
# 目标/真实标签
# 它为每一个时刻后的值，因为这才是下一个时刻的值。
# Creating lists that will hold our input and target sequences
input_seq = []
target_seq = []
 
for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])
 
    # Remove first character for target sequence
    target_seq.append(text[i][1:])
    print("Input Sequence: {}\nTarget Sequence: {}".format(input_seq[i], target_seq[i]))
 
# 现在我们可以通过使用上面创建的字典映射输入和目标序列到整数序列。
# 这将允许我们随后对输入序列进行一次one-hot encoding。
 
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]
 
# 定义如下三个变量
# dict_size: 字典的长度，即唯一字符的个数。它将决定one-hot vector的长度
# seq_len:输入到模型中的sequence长度。这里是最长的句子的长度-1，因为不需要最后一个字符
# batch_size: mini batch的大小，用于批量训练
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)
 
 
def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)
 
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features
# 同时定义一个helper function，用于初始化one-hot向量
# Input shape --> (Batch Size, Sequence Length, One-Hot Encoding Size)
input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
 
# 到此我们完成了所有的数据预处理，可以将数据从NumPy数组转为PyTorch张量啦
input_seq = torch.from_numpy(input_seq)
target_seq = torch.Tensor(target_seq)
 
# 接下来就是搭建模型的步骤，你可以在这一步使用全连接层，卷积层，RNN层，LSTM层等等。
# 但是我在这里使用最最基础的nn.rnn来示例一个RNN是如何使用的。
from RNN_model import Model
 
"""
# 在开始构建模型之前，让我们使用 PyTorch 中的内置功能来检查我们正在运行的设备（CPU 或 GPU）。
# 此实现不需要 GPU，因为训练非常简单。
# 但是，随着处理具有数百万个可训练参数的大型数据集和模型，使用 GPU 对加速训练非常重要。
# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
# is_cuda = torch.cuda.is_available()
# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")
"""
 
 
# 要开始构建我们自己的神经网络模型，我们可以为所有神经网络模块定义一个继承 PyTorch 的基类（nn.module）的类。
# 这样做之后，我们可以开始在构造函数下定义一些变量以及模型的层。 对于这个模型，我们将只使用一层 RNN，然后是一个全连接层。 全连接层将负责将 RNN 输出转换为我们想要的输出形状。
# 我们还必须将 forward() 下的前向传递函数定义为类方法。 前向函数是按顺序执行的，因此我们必须先将输入和零初始化隐藏状态通过 RNN 层，然后再将 RNN 输出传递到全连接层。 请注意，我们使用的是在构造函数中定义的层。
# 我们必须定义的最后一个方法是我们之前调用的用于初始化hidden state的方法 - init_hidden()。 这基本上会在我们的隐藏状态的形状中创建一个零张量。
 
 
 
# 在定义了上面的模型之后，我们必须用相关参数实例化模型并定义我们的超参数。 我们在下面定义的超参数是：
# n_epochs: 模型训练所有数据集的次数
# lr: learning rate学习率
 
# 与其他神经网络类似，我们也必须定义优化器和损失函数。 我们将使用 CrossEntropyLoss，因为最终输出基本上是一个分类任务和常见的 Adam 优化器。
# Instantiate the model with hyperparameters
model = Model(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)
# We'll also set the model to the device that we defined earlier (default is CPU)
# model.to(device)
# Define hyperparameters
n_epochs = 100 # 训练轮数
lr = 0.01 # 学习率
# Define Loss, Optimizer
loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 采用Adam作为优化器
# 现在我们可以开始训练了！
# 由于我们只有几句话，所以这个训练过程非常快。
# 然而，随着我们的进步，更大的数据集和更深的模型意味着输入数据要大得多，并且我们必须计算的模型中的参数数量要多得多。
# Training Run
for epoch in range(1, n_epochs + 1):
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    # input_seq.to(device) # 使用GPU
    output, hidden = model(input_seq)
    loss = loss_fn(output, target_seq.view(-1).long())
    loss.backward()  # Does backpropagation and calculates gradients
    optimizer.step()  # Updates the weights accordingly
    if epoch % 10 == 0:
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))