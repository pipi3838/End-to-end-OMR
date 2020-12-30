import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, voc_len=1781, conv_blocks=4):
        super(CRNN, self).__init__()
        
        width_reduction = 1
        height_reduction = 1       
        self.conv_blocks = conv_blocks
        self.conv_filter_n = [32, 64, 128, 256]
        self.conv_filter_size = [ 3, 3, 3, 3 ]
        self.conv_pooling_size = [ 2, 2, 2, 2 ]
        self.rnn_uints = 512
        self.rnn_layers = 2
        self.rnn_prob = 0.5
        self.img_height = 128
        self.voc_len = voc_len

        #CNN
        layers = []
        for i in range(conv_blocks):
            if i == 0:
                layers.append(nn.Conv2d(1, self.conv_filter_n[i], kernel_size=self.conv_filter_size[i], padding=1))
            else:
                layers.append(nn.Conv2d(self.conv_filter_n[i-1], self.conv_filter_n[i], kernel_size=self.conv_filter_size[i], padding=1))
                
            layers.append(nn.BatchNorm2d(self.conv_filter_n[i]))
            layers.append(nn.LeakyReLU())
            layers.append(nn.MaxPool2d(kernel_size=self.conv_pooling_size[i], stride=self.conv_pooling_size[i]))
            width_reduction = width_reduction * self.conv_pooling_size[i]
            height_reduction = height_reduction * self.conv_pooling_size[i]
        self.cnn = nn.Sequential(*layers) 
        
        feature_dim = int(self.conv_filter_n[-1]*self.img_height/height_reduction)
        # self.multiheadattn = nn.MultiheadAttention(feature_dim, num_heads = 2, dropout=0.3)
        #RNN
        self.rnn = nn.LSTM(feature_dim, self.rnn_uints, self.rnn_layers, dropout=self.rnn_prob, bidirectional=True)
        self.fc = nn.Linear(self.rnn_uints*2, self.voc_len+1)  #把 hidden state 線性轉換成 output
        
        
    def forward(self,x):
        out = self.cnn(x)  #[bs, 256, 8, w]
        #prepare for rnn
        out = out.permute(3,0,2,1)  # [width(sequence_len), bs, height, channels] = [w, bs(12), h(8), c(256)]
        feature_dim = out.shape[2] * self.conv_filter_n[-1]  # channel * 輸出高度
        out = out.reshape(out.shape[0], x.shape[0], feature_dim)  #[w, bs, 256*8(2048)]
        out,_ = self.rnn(out)  #[w, bs, 1024(units*directions)]
        
        T, b, h = out.size()
        out = out.view(T*b, h)
        out = self.fc(out) #[w, bs, 1782]
        out = out.view(T, b, -1)
        out = F.log_softmax(out, dim=2)
        # out, attn_w = self.multiheadattn(out, out, out)
        # out,_ = self.rnn(out)  #[w, bs, 1024(units*directions)]
        # out = self.fc(out) #[w, bs, 1782]
        # out = F.log_softmax(out, dim=2)

        return out
    


# model = CRNN()
# #print(model)
# inputx = torch.randn(16,1,128,1500)
# output = model(inputx)
# print(output.shape) #torch.Size([w,12,1782])

