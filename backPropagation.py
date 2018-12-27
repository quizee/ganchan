import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = Variable(torch.Tensor([1.0]),  requires_grad=True)  # Any random value
w2 = Variable(torch.Tensor([1.0]),  requires_grad=True)
b= 1.0

def forward(x):
    return x*x*w2 + x*w1 + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

# Before training
print("predict (before training)",  4, forward(4).data[0])


# Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)
        l.backward()
        print("\tgrad: ", x_val, y_val, w1.grad.data[0], w2.grad.data[0])
        #update(step)
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        w1.grad.data.zero_()
        w2.grad.data.zero_()

    print("progress:", epoch, l.data[0])

#After training
print("predict (after training)",  4, forward(4).data[0])
