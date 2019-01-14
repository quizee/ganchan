from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

'''전체적인 순서를 정리해보자면
1) 데이터불러오기 단계 (dataset --> dataloader)
   train_loader/ test_loader 분리
2) model 설계 단계
   class model
   __init__ 함수: 각 linear함수 정의 --> self.l = torch.nn.Linear(입력개수, 출력개수)
   forward 함수: 정의한 각 linear 함수를 activation funtion으로 연결한다.
3) 설계한 model에 대한 객체 생성 --> 객체의 여러 메소드를 활용한다
4) criterion, optimizer 정의하기
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
5) train단계에서 criterion으로 forward pass, optimizer로 backward pass
6) test단계에서는 loss 계산 + target 과 비교
   (모델을 구축하는 단계가 아니므로 backward 는 할 필요가 없다.)
  '''
#Training settings
batch_size = 64

#MNIST Dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True,
                               transform=transfroms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False,
                              transform=transforms.ToTensor())

#Data Loader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

#model 설계에서 뭐해줘야되지?
#각 linear 정의해주기 self.l = nn.Linear(입력개수, 출력개수)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)

model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01, momentum =0.5)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        #기본 설정
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        #forward pass
        output = model(data)
        loss = criterion(output, target)
        #backward pass
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
def test():
    model.eval()
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        optimizer.zero_grad()
        #forward pass: 테스트 단계에서는 backward를 통해 모델을 형성할 필요가 없다
        output = model(data)
        total_loss += criterion(output, target).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, 10):
    train(epoch)
    test()
