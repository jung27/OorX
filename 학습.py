# import torch
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms

# train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
# test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)


# # Define transform
# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,), (0.5,))
#             ])
 
# # Load training set of MNIST dataset
# train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# for t in train_set:
#     image, label = t
#     image[0]
import 합성곱
from 오차역전파법 import Variable
import math
import numpy as np

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

# import random

# def rand_weight(mean=0.0, std=0.5):
#     return Variable(random.gauss(mean, std))

# # 은닉층 3개
# hidden1 = [[rand_weight() for _ in range(4)] for _ in range(4)]
# hidden2 = [[rand_weight() for _ in range(4)] for _ in range(4)]
# hidden3 = [[rand_weight() for _ in range(4)] for _ in range(4)]

# # bias (thresholds)
# threshold1 = rand_weight(0.0, 0.1)
# threshold2 = rand_weight(0.0, 0.1)
# threshold3 = rand_weight(0.0, 0.1)

# # 출력층 가중치: 3 x 3 크기 필터 * 3개 필터 + bias
# output_weights1 = [
#     [[rand_weight() for _ in range(3)] for _ in range(3)],
#     [[rand_weight() for _ in range(3)] for _ in range(3)],
#     [[rand_weight() for _ in range(3)] for _ in range(3)],
#     rand_weight(0.0, 0.1)
# ]

# output_weights2 = [
#     [[rand_weight() for _ in range(3)] for _ in range(3)],
#     [[rand_weight() for _ in range(3)] for _ in range(3)],
#     [[rand_weight() for _ in range(3)] for _ in range(3)],
#     rand_weight(0.0, 0.1)
# ]

# Hidden Layer 1
hidden1 = [
    [Variable(0.2681589691589437), Variable(-0.6998247166092413), Variable(-0.23962026428962951), Variable(-0.3345726472901726)],
    [Variable(0.3298563215890688), Variable(-0.08662972402929989), Variable(-0.3321056327460546), Variable(-0.4598000852387601)],
    [Variable(0.6346519604171993), Variable(1.2949914903329092), Variable(-0.7277966722191559), Variable(-1.284372591047918)],
    [Variable(0.006313452416890661), Variable(0.5959712741688549), Variable(0.49853813831746197), Variable(0.21085862874539715)]
]

# Hidden Layer 2
hidden2 = [
    [Variable(0.0827089731512474), Variable(-0.5555834258364883), Variable(-0.9596336833340979), Variable(-0.18293328326725738)],
    [Variable(0.8258771242624293), Variable(1.596609616473133), Variable(-0.36289062755432416), Variable(-1.3516417330259742)],
    [Variable(-0.4216688453280998), Variable(1.3406011879792303), Variable(0.7594110950879972), Variable(-1.5155543350085836)],
    [Variable(-1.8997606535496752), Variable(0.4489121580929547), Variable(1.0560192812628109), Variable(0.7340086316382727)]
]

# Hidden Layer 3
hidden3 = [
    [Variable(-1.0306669672719297), Variable(-0.21784271791158788), Variable(-0.35115536411722975), Variable(0.8576878743824181)],
    [Variable(0.9172063946934682), Variable(0.4061341232535921), Variable(0.31091582257219075), Variable(-0.6082627781487051)],
    [Variable(1.3824425718057844), Variable(-1.2169641303564755), Variable(-0.003395700948247735), Variable(0.57222583046061)],
    [Variable(1.18963124580428), Variable(-0.011205908641844488), Variable(-0.6878507873621724), Variable(0.1157070332701488)]
]

# Thresholds (Biases)
threshold1 = Variable(0.33953765718522017)
threshold2 = Variable(1.375441171218558)
threshold3 = Variable(0.8123497756518673)

# Output Weights for Class 1
output_weights1 = [
    [
        [Variable(0.3767949818613217), Variable(-0.530915315071232), Variable(0.42262782225452633)],
        [Variable(1.0093936015482325), Variable(-0.8287968938551392), Variable(-0.036967025732950874)],
        [Variable(-0.5161572663622789), Variable(1.4595200519612508), Variable(-0.46809570329103045)]
    ],
    [
        [Variable(-1.113532345074961), Variable(-0.4602668097070901), Variable(1.9289339518593331)],
        [Variable(1.197178984542263), Variable(-1.1994527460856885), Variable(0.14066198694921925)],
        [Variable(0.7595217091333265), Variable(-1.0222553578661104), Variable(-0.4104543724036289)]
    ],
    [
        [Variable(0.42479515416851105), Variable(0.6178694093143158), Variable(-0.05778654397162481)],
        [Variable(0.02220446734600164), Variable(1.2977113706166723), Variable(-1.5167669168558873)],
        [Variable(-0.7737005860888172), Variable(-0.4530557758195354), Variable(0.007899983241867916)]
    ],
    Variable(0.05603059561325925)
]

# Output Weights for Class 2
output_weights2 = [
    [
        [Variable(-0.04147799723839537), Variable(0.23354585812105821), Variable(-0.0461212415894532)],
        [Variable(-0.7569756861442718), Variable(1.3648834465113842), Variable(0.11517415709368753)],
        [Variable(-1.4235597481204216), Variable(0.5332866099484533), Variable(0.7402138250048043)]
    ],
    [
        [Variable(0.284487885165875), Variable(0.29175965443474955), Variable(-2.675272344932545)],
        [Variable(-1.49864352508825), Variable(1.3292337986962288), Variable(-0.6638172246731455)],
        [Variable(-0.5554011854203207), Variable(-0.6152758831124254), Variable(1.331936345736745)]
    ],
    [
        [Variable(-1.766118596257446), Variable(-0.4780044638171114), Variable(0.5895413641438858)],
        [Variable(0.014241584579218213), Variable(-0.07960242988588836), Variable(1.1643018737262023)],
        [Variable(0.44946113550528954), Variable(-0.22984078346710288), Variable(-0.2127538164532074)]
    ],
    Variable(-0.03224634860008267)
]

for j in range(5):
    for i in range(len(images)):
        test_image = images[i]
        first = 합성곱.convolution(test_image, hidden1, threshold1)
        second = 합성곱.convolution(test_image, hidden2, threshold2)
        third = 합성곱.convolution(test_image, hidden3, threshold3)

        max_pool1 = max_pooling(first, pool_size=2, stride=None)
        max_pool2 = max_pooling(second, pool_size=2, stride=None)
        max_pool3 = max_pooling(third, pool_size=2, stride=None)

        result1 = 합성곱.sumproduct(max_pool1, output_weights1[0], (0, 0)) + \
                합성곱.sumproduct(max_pool2, output_weights1[1], (0, 0)) + \
                합성곱.sumproduct(max_pool3, output_weights1[2], (0, 0)) + (-1 * output_weights1[3])
        output1 = 1 / (1 + math.e ** (-1 * result1))

        result2 = 합성곱.sumproduct(max_pool1, output_weights2[0], (0, 0)) + \
                합성곱.sumproduct(max_pool2, output_weights2[1], (0, 0)) + \
                합성곱.sumproduct(max_pool3, output_weights2[2], (0, 0)) + (-1 * output_weights2[3])
        output2 = 1 / (1 + math.e ** (-1 * result2))

        label = labels[i]  # 0 또는 1
        target1 = 1 - label  # label이 0이면 target1=1 (1일 확률이 높아야), label이 1이면 target1=0
        target2 = label      # label이 0이면 target2=0 (2일 확률 낮아야), label이 1이면 target2=1

        loss = ((output1 + -1 * target1) ** 2 + (output2 + -1 * target2) ** 2)

        # if loss.data < 0.1:
        #     continue

        print(f"Image {i+1}:")
        print(f"  O일 확률: {output1.data*100:.2f}%")
        print(f"  X일 확률: {output2.data*100:.2f}%")
        print(f"  정답: {labels[i]}")
        print(f"  오차: {loss.data}")
        loss.backward()

        #경사하강법
        learning_rate = 0.01

        for h in hidden1:
            for v in h:
                v.data -= learning_rate * v.grad
        for h in hidden2:
            for v in h:
                v.data -= learning_rate * v.grad
        for h in hidden3:
            for v in h:
                v.data -= learning_rate * v.grad

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