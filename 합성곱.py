# ===== 합성곱.py =====
from 오차역전파법 import Variable

def sumproduct(image, kernel, pos):
    out = Variable(0.0)
    K = len(kernel)
    for i in range(K):
        for j in range(K):
            out = out + image[pos[1]+i][pos[0]+j] * kernel[i][j]
    return out

def convolution(image, kernel, bias):
    H,W = len(image), len(image[0])
    K   = len(kernel)
    out = []
    for y in range(H-K+1):
        row = []
        for x in range(W-K+1):
            sp = sumproduct(image, kernel, (x,y))
            row.append(sp - bias)
        out.append(row)
    return out

def max_pooling(matrix, pool_size=2, stride=None):
    if stride is None: stride = pool_size
    H,W = len(matrix), len(matrix[0])
    oh = (H-pool_size)//stride + 1
    ow = (W-pool_size)//stride + 1
    pooled = [[None]*ow for _ in range(oh)]
    for i in range(oh):
        for j in range(ow):
            win = []
            for di in range(pool_size):
                for dj in range(pool_size):
                    win.append(matrix[i*stride+di][j*stride+dj])
            m = win[0]
            for v in win[1:]:
                if v.data > m.data: m = v
            pooled[i][j] = m
    return pooled