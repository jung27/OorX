import math

def sumproduct(image, kernel, pos):
    size = len(kernel)
    output = 0
    for i in range(size):
        for j in range(size):
            value = image[pos[1] + i][pos[0] + j] * kernel[i][j]
            output = output + value
    return output

def convolution(image, kernel, bias=0):
    output = []
    image_height = len(image)
    image_width = len(image[0])
    kernel_size = len(kernel)
    
    for y in range(image_height - kernel_size + 1):
        row = []
        for x in range(image_width - kernel_size + 1):
            value = sumproduct(image, kernel, (x, y))
            # Apply the sigmoid function
            sigmoid_value = 1 / (1 + math.e ** (-1 * value + bias))
            row.append(sigmoid_value)
        output.append(row)
    
    return output