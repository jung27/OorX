import 합성곱
from 오차역전파법 import Variable
import math
import numpy as np
import random

# 파일 경로
images = np.load("images.npy")
labels = np.load("labels.npy")
# images = np.load("images_from_excel.npy")
# labels = np.load("labels_from_excel.npy")

def max_pooling(matrix, pool_size=2, stride=None):
    if stride is None:
        stride = pool_size

    h = len(matrix)
    w = len(matrix[0])
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1

    # 출력 행렬 초기화
    output = [[None] * out_w for _ in range(out_h)]

    for i in range(out_h):
        for j in range(out_w):
            # 윈도우 내 모든 값 중 최댓값 선택
            window_vals = []
            for di in range(pool_size):
                for dj in range(pool_size):
                    window_vals.append(matrix[i*stride + di][j*stride + dj])
            output[i][j] = max(window_vals)
    return output

def softmax(a: Variable, b: Variable):
    e = math.e
    return (e ** a / (e ** a + e ** b), e ** b / (e ** a + e ** b))

def rand_weight(mean=0.0, std=0.1):
    return Variable(random.gauss(mean, std))

# 은닉층 3개
hidden1 = [[rand_weight() for _ in range(4)] for _ in range(4)]
hidden2 = [[rand_weight() for _ in range(4)] for _ in range(4)]
hidden3 = [[rand_weight() for _ in range(4)] for _ in range(4)]

# bias (thresholds)
threshold1 = rand_weight()
threshold2 = rand_weight()
threshold3 = rand_weight()

# 출력층 가중치: 3 x 3 크기 필터 * 3개 필터 + bias
output_weights1 = [
    [[rand_weight() for _ in range(3)] for _ in range(3)],
    [[rand_weight() for _ in range(3)] for _ in range(3)],
    [[rand_weight() for _ in range(3)] for _ in range(3)],
    rand_weight()
]

output_weights2 = [
    [[rand_weight() for _ in range(3)] for _ in range(3)],
    [[rand_weight() for _ in range(3)] for _ in range(3)],
    [[rand_weight() for _ in range(3)] for _ in range(3)],
    rand_weight()
]

for j in range(10):
    indices = list(range(len(images)))
    random.shuffle(indices)
    for i in indices:  # 10개 이미지에 대해서만 학습
        test_image = images[i]
        first = 합성곱.convolution(test_image, hidden1, threshold1)
        second = 합성곱.convolution(test_image, hidden2, threshold2)
        third = 합성곱.convolution(test_image, hidden3, threshold3)

        max_pool1 = max_pooling(first, pool_size=2, stride=None)
        max_pool2 = max_pooling(second, pool_size=2, stride=None)
        max_pool3 = max_pooling(third, pool_size=2, stride=None)

        result1 = (합성곱.sumproduct(max_pool1, output_weights1[0], (0, 0)) + \
                합성곱.sumproduct(max_pool2, output_weights1[1], (0, 0)) + \
                합성곱.sumproduct(max_pool3, output_weights1[2], (0, 0)) + (-1 * output_weights1[3])).sigmoid()

        result2 = (합성곱.sumproduct(max_pool1, output_weights2[0], (0, 0)) + \
                합성곱.sumproduct(max_pool2, output_weights2[1], (0, 0)) + \
                합성곱.sumproduct(max_pool3, output_weights2[2], (0, 0)) + (-1 * output_weights2[3])).sigmoid()

        output1, output2 = softmax(result1, result2)

        label = [0, 0]
        label[labels[i]] = 1

        loss = -label[0] * output1.log() - label[1] * output2.log()

        print(f"Image {i+1}:")
        # print(f"  O일 확률: {output1.data*100:.2f}%")
        # print(f"  X일 확률: {output2.data*100:.2f}%")
        print(f"  정답: {labels[i]}")
        print(f"  오차: {loss.data}")
        loss.backward()
        #경사하강법
        learning_rate = 0.01

        for h in hidden1:
            for v in h:
                v.data -= learning_rate * v.grad
                v.zero_grad()
        for h in hidden2:
            for v in h:
                v.data -= learning_rate * v.grad
                v.zero_grad()
        for h in hidden3:
            for v in h:
                v.data -= learning_rate * v.grad
                v.zero_grad()

        threshold1.data -= learning_rate * threshold1.grad
        threshold2.data -= learning_rate * threshold2.grad
        threshold3.data -= learning_rate * threshold3.grad
        threshold1.zero_grad()
        threshold2.zero_grad()
        threshold3.zero_grad()

        for w in output_weights1:
            if isinstance(w, Variable):
                w.data -= learning_rate * w.grad
                w.zero_grad()
                continue
            for v in w:
                for c in v:
                    if isinstance(c, Variable):
                        c.data -= learning_rate * c.grad
                        c.zero_grad()
        for w in output_weights2:
            if isinstance(w, Variable):
                w.data -= learning_rate * w.grad
                w.zero_grad()
                continue
            for v in w:
                for c in v:
                    if isinstance(c, Variable):
                        c.data -= learning_rate * c.grad
                        c.zero_grad()


print(f"  Hidden1: {hidden1}")
print(f"  Hidden2: {hidden2}")
print(f"  Hidden3: {hidden3}")
print(f"  Threshold1: {threshold1}")
print(f"  Threshold2: {threshold2}")
print(f"  Threshold3: {threshold3}")
print(f"  Output Weights1: {output_weights1}")
print(f"  Output Weights2: {output_weights2}")