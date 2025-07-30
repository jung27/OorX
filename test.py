# from model_utils import predict

# prob = predict([[0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 0, 0, 1, 0, 1],
#                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0, 0, 0, 0, 0]], "model.npy")
# print("O일 확률:", 1- prob)

import json
import numpy as np

# 1) model.npy 불러오기
model = np.load("model.npy", allow_pickle=True).item()
# model = {
#   "conv_kernels": [  np.array 8×3×3, ... ],
#   "conv_biases":  [ 8 floats ],
#   "fc_w":         [ feat_dim floats ],
#   "fc_b":         float
# }

# 2) 리스트 변환
json_model = {
    "conv_kernels": [
        [ list(row)           # 한 행(row)을 리스트로
          for row in kernel  # 3×3 kernel 의 각 행
        ]
        for kernel in model["conv_kernels"]  # 8개의 채널
    ],
    "conv_biases": list(model["conv_biases"]),
    "fc_w":        list(model["fc_w"]),
    "fc_b":        float(model["fc_b"])
}

# 3) 직렬화
with open("model.json", "w") as f:
    json.dump(json_model, f)
print("✅ model.json 저장 완료")