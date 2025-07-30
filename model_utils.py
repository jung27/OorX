# model_utils.py

import numpy as np
import math
import 합성곱
from 오차역전파법 import Variable

def load_model(path="model.npy"):
    """
    np.save로 저장된 dict를 load하여 반환합니다.
    {
      "conv_kernels": list of 8×3×3 floats,
      "conv_biases":  list of 8 floats,
      "fc_w":         list of feat_dim floats,
      "fc_b":         float
    }
    """
    data = np.load(path, allow_pickle=True).item()
    return data

def predict(image: np.ndarray, model_path="model.npy") -> float:
    """
    단일 9×9 바이너리 이미지에 대해 학습된 모델을 실행하고
    0~1 사이의 확률(prediction)을 반환합니다.
    
    image: numpy.ndarray, shape (9,9), 값 0 or 1
    model_path: 저장된 model.npy 경로
    """
    # 1) 모델 불러오기
    mdl = load_model(model_path)
    conv_kernels = mdl["conv_kernels"]   # shape [C,3,3]
    conv_biases  = mdl["conv_biases"]    # length C
    fc_w         = mdl["fc_w"]           # length feat_dim
    fc_b         = mdl["fc_b"]           # scalar

    C = len(conv_kernels)
    
    # 2) Variable 객체로 래핑
    img_var = [[Variable(float(p)) for p in row] for row in image]
    kern_vars = [
        [[Variable(conv_kernels[c][i][j]) for j in range(3)] for i in range(3)]
        for c in range(C)
    ]
    bias_vars = [Variable(b) for b in conv_biases]
    fc_w_vars  = [Variable(w) for w in fc_w]
    fc_b_var   = Variable(fc_b)

    # 3) Conv → ReLU → MaxPool → Flatten
    feats = []
    for kern, b in zip(kern_vars, bias_vars):
        fm = 합성곱.convolution(img_var, kern, b)          # 7×7
        fm = [[v.relu() for v in row] for row in fm]       # ReLU
        pm = 합성곱.max_pooling(fm, pool_size=2)           # 3×3
        for row in pm:
            feats.extend(row)                              # flatten

    # 4) FC → Sigmoid
    logit = Variable(0.0)
    for w_var, x_var in zip(fc_w_vars, feats):
        logit = logit + w_var * x_var
    logit = logit + fc_b_var
    pred = logit.sigmoid()

    return pred.data