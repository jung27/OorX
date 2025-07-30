# ===== 학습.py =====
import math, random
import numpy as np
from 오차역전파법 import Variable
import 합성곱

# —— Hyperparams —— #
epochs     = 50
batch_size = 32
init_lr    = 1e-2
beta1      = 0.9
beta2      = 0.999
eps        = 1e-8
lr_decay   = 0.95

# —— Data —— #
images = np.load("images.npy")  # (N,9,9)
labels = np.load("labels.npy")  # 0 or 1
N,H,W = images.shape

# —— Model Setup —— #
def xavier(f_in,f_out):
    lim = math.sqrt(6/(f_in+f_out))
    return random.uniform(-lim, lim)

C = 8  # conv channels
# Conv kernels + biases
conv_kernels = [
    [[Variable(xavier(9,9)) for _ in range(4)] for _ in range(4)]
    for _ in range(C)
]
conv_biases = [Variable(0.0) for _ in range(C)]

# After conv(4×4)->relu->pool(2×2): feature map size = 3×3
feat_h = (H-4+1 - 2)//2 +1
feat_w = (W-4+1 - 2)//2 +1
feat_dim = C * feat_h * feat_w

# FC layer
fc_w = [Variable(xavier(feat_dim,1)) for _ in range(feat_dim)]
fc_b = Variable(0.0)

# Collect all params
params = []
for kern in conv_kernels:
    for row in kern:
        params += row
params += conv_biases
params += fc_w + [fc_b]

# Adam state
m = {p:0.0 for p in params}
v = {p:0.0 for p in params}
t = 0

# —— Training Loop —— #
indices = list(range(N))
lr = init_lr

for ep in range(1, epochs+1):
    total_loss = 0.0
    random.shuffle(indices)

    for b in range(0, N, batch_size):
        batch_idx = indices[b:b+batch_size]
        # accumulate batch loss
        batch_loss = Variable(0.0)

        for i in batch_idx:
            # 1) to Variable image
            img = [[Variable(px) for px in row] for row in images[i]]

            # 2) conv→relu→pool→flatten
            feats = []
            for kern, cb in zip(conv_kernels, conv_biases):
                fm = 합성곱.convolution(img, kern, cb)
                fm = [[v.relu() for v in row] for row in fm]
                pm = 합성곱.max_pooling(fm,2)
                for row in pm: feats.extend(row)

            # 3) FC(logit)→sigmoid
            logit = Variable(0.0)
            for w,x in zip(fc_w, feats):
                logit = logit + w * x
            logit = logit + fc_b
            pred  = logit.sigmoid()

            # 4) BCE loss
            yv   = Variable(float(labels[i]))
            loss = -( yv*pred.log()
                    + (1.0-yv)*(1.0-pred).log() )
            batch_loss = batch_loss + loss
            total_loss += loss.data

        # 5) backward & Adam update
        batch_loss = batch_loss / len(batch_idx)
        batch_loss.backward()
        t += 1

        for p in params:
            g = p.grad
            m[p] = beta1*m[p] + (1-beta1)*g
            v[p] = beta2*v[p] + (1-beta2)*(g*g)
            m_hat = m[p] / (1 - beta1**t)
            v_hat = v[p] / (1 - beta2**t)
            p.data -= lr * m_hat / (math.sqrt(v_hat) + eps)

        batch_loss.zero_grad()

    lr *= lr_decay
    print(f"Epoch {ep:2d}/{epochs} — avg loss: {total_loss/N:.4f}")

# Save model params
# Variable 객체에서 실제 수치만 추출
model = {
    "conv_kernels": [
        [ [v.data for v in row] for row in kern ]
        for kern in conv_kernels
    ],
    "conv_biases": [ b.data for b in conv_biases ],
    "fc_w":         [ w.data for w in fc_w ],
    "fc_b":         fc_b.data
}

# numpy 파일로 저장 (pickle 기반)
np.save("model.npy", model)
print("📦 모델 파라미터를 model.npy에 저장했습니다.")