import torch
#import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

xy = np.loadtxt('diabetes.csv',delimiter=',', dtype=np.float32)

x_data = Variable(torch.from_numpy(xy[:,0:-1]))
y_data = Variable(torch.from_numpy(xy[:,[-1]]))

class Model(torch.nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        self.sigmoid = torch.nn.Sigmoid()

    #y^을 정의하는 부분
    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

model = Model()

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr= 0.1)

for epoch in range(100):
    #forward pass
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#after Training
hour_var = Variable(torch.Tensor([[0.1,0.5,0,-0.3,0,0.01,-0.5,-0.2]]))
print("predict", model(hour_var).data[0][0] > 0.5)
