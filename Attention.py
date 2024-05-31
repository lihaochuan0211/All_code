import torch
from torch import nn
import torch.nn.functional as F
import math

X = torch.rand(128, 64, 512)  # batch, time, dimension
print(X.shape)
d_model = 512
n_head = 8

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__() # 参数初始化

        self.n_head = n_head
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)  # 把初始函数映射到Q K V
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)  # 由于多头注意力机制 要组合映射
        self.softmax = nn.Softmax(dim=2) # Softmax 

    def forward(self, q, k, v, mask=None):
        batch, time, dimension = q.shape   #（128，64，512）
        n_d = self.d_model // self.n_head  # 每个头的一个新维度  512/8=64 每个头的维度
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) # 映射  512->512

        q = q.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)  # 重塑 128,64,512 -> 128,64,8,64(batch, time, self.n_head, n_d)-> 128,8,64,64 (batch, self.n_head, time, n_d)
        k = k.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, time, self.n_head, n_d).permute(0, 2, 1, 3)

        score = q @ k.transpose(2, 3) / math.sqrt(n_d) # Q K  
        if mask is not None:
            mask = torch.tril(torch.ones(time, time, dtype=bool)) # 下三角矩阵
            score = score.masked_fill(mask == 0, float('-inf')) # Mask
        score = self.softmax(score) @ v  # Softmax

        score = score.permute(0, 2, 1, 3).contiguous().view(batch, time, dimension) # Concat

        output = self.w_combine(score)  # 映射
        return output
    

attention = MultiHeadAttention(d_model, n_head)
output = attention(X, X, X)
print(output.shape)
print(output)