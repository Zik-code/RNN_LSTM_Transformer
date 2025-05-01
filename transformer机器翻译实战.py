











# 构建代码关键的两点：
# 1. 从整体到局部，先搭建大框架
# 2. 搞清整个模型的流动中形状的改变


class EncoderLayer(nn.Module):


class DecoderLayer(nn.Module)  :

class Transformer(nn.Model):











if __name__ == "__main__":
    

    sentences = ['ich mochte ein bier P','S i want a beer','i want a beer E']
    src_vocab = {'P':0,'ich':1,'mochte':2,'ein':3,'bier':4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P':0,'i':1,'want':2,'a':3,'beer':4,'S':5,'E':6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5
    tgt_len = 5

    d_model = 512
    


# 一个batch里面的句子长度可能不同，预设一个值，超过这个值长度的删除，不足的填充
