'''
선형회귀분석 예제 _ 파이토치에서 딥러닝을 수행하는 과정 요약
1. nn.Module 클래스를 상속받아 (forward 함수를 통해) 모델 구조 클래스 선언
2. 해당 클래스 객체 생성
3. SGD나 Adam 등의 옵티마이저를 생성하고, 생성한 모델의 파라미터를 최적화 대상으로 등록
4. 데이터로 미니배치를 구성하여 피드포워드 연산 그래프 생성
5. 손실 함수를 통해 최종 결과값(scalar)과 손실값(loss) 계산
6. 손실에 대해 backward() 호출 -> 연산 그래프 상의 텐서들의 기울기가 채워짐
7. 3번의 옵티마이저에서 step()을 호출하여 경사하강법 1스텝 수행
8. 4번으로 돌아가 수렴 조건이 만족할 때까지 반복 수행
'''

import random
import torch
import torch.nn as nn

# 1. 1개의 선형 계층을 가진 MyModel 모듈 선언
class MyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self, x):
        y = self.linear(x)
        return y

# 2. y = 3x1 + x2 -2x3 로 동작한다고 가정하고, 이 함수를 근사로 맞춰라
def ground_truth(x):
    return 3 * x[:,0] + x[:,1] -2 * x[:,2]

# 3. 모델과 텐서 입력 -> 피드포워딩 -> 역전파 -> 경사하강법 한 스텝 수행하는 함수 정의
def train(model, x, y, optim):
    optim.zero_grad()

    y_hat = model(x) # feed-forward
    loss = ((y - y_hat)**2).sum() / x.size(0) # get error

    loss.backward()

    optim.step() # one-step of gradient descent

    return loss.data

# 4. 앞의 함수를 사용하기 위한 하이퍼파라미터 설정
batch_size = 1
n_epochs = 1000
n_iter = 10000

model = MyModel(3,1)
optim = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.1)

print(model)

# 5. 평균 손실, 에러가 0.001보다 작아질 때까지 훈련 진행.
for epoch in range(n_epochs):
    avg_loss = 0

    for i in range(n_iter):
        x = torch.rand(batch_size, 3)
        y = ground_truth(x.data)
        loss = train(model,x,y,optim)

        avg_loss += loss
        avg_loss = avg_loss / n_iter

    # simple test sample to check the network.
    x_valid = torch.FloatTensor([[0.3, 0.2, 0.1]])
    y_valid = ground_truth(x_valid)

    model.eval()
    y_hat = model(x_valid)
    model.train()

    print(avg_loss, y_valid.data[0], y_hat.data[0,0])

    if avg_loss < 0.001:
        break