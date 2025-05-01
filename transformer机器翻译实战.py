











# 构建代码关键的两点：
# 1. 从整体到局部，先搭建大框架
# 2. 搞清整个模型的流动中形状的改变


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        



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
    # 前馈神经网络linear层的输出维度
    d_ff = 2048
    d_k = d_v =64
    # 编码器解码器和堆叠的层数
    n_layers = 6
    # 注意力头数
    n_heads = 8

    model = Transformer()

    


# 一个batch里面的句子长度可能不同，预设一个值，超过这个值长度的删除，不足的填充
