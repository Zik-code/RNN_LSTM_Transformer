# test.py
# 现在让我们测试我们的模型，看看我们会得到什么样的输出。 作为第一步，我们将定义一些辅助函数来将我们的模型输出转换回文本。
# This function takes in the model and character as arguments and returns the next character prediction and hidden state
import numpy as np
import torch
from torch import device
import torch.nn as nn
from train import char2int, one_hot_encode, dict_size, int2char, model
 
 
def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)
    # character.to(device)
 
    out, hidden = model(character)
 
    prob = nn.functional.softmax(out[-1], dim=0).data
 
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()
 
    return int2char[char_ind], hidden
 
# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, chars)
        chars.append(char)
 
    return ''.join(chars)