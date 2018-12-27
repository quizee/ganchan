import  numpy as np
import matplotlib.pyplot as plt
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

a = 0.0
b = 3.0
c = 1.0

#our model for the forward pass
def forward(x):
    return x*x*a + x*b + c

def loss(x, y):
    y_pred = forward(x)
    return (y - y_pred) * (y - y_pred)

#gradient about a
def gradient1(x, y):
    return 2*x*x * (x*x*a - y + b*x + c)

#gradient about b
def gradient2(x, y):
    return 2*x * (x*b - y + a*x*x + c)

#before training
print("predict (before training)", 4, forward(4))

#training loop
for epoch in range(100):
    for x_val, y_val in zip(x_data, y_data):
        grad1 = gradient1(x_val, y_val)
        grad2 = gradient2(x_val, y_val)
        #gradient 먼저 계산하고
        a = a - 0.01 * grad1
        b = b - 0.01 * grad2
        print("\tgrad: ",x_val, y_val, grad1, grad2)
        l = loss(x_val, y_val)

    print("progress: ", epoch, "a= ", a, "b= ",b, "loss", l)

#after training
print("predict (after training)", "4 hours", forward(4))
