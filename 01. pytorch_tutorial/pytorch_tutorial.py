import torch
import numpy as np

# 1. numpy 와 torch.Tensor 비교 =============================
x = torch.Tensor([[1,2],[3,4]])
x = torch.from_numpy(np.array([[1,2],[3,4]]))
x = np.array([[1,2],[3,4]])
#print(x)

# 2. 자동미분 및 역전파를 수행하는 autograd ======================
x = torch.FloatTensor(2,2)
y = torch.FloatTensor(2,2)
y.requires_grad_(True)

'''
텐서들 간에 연산을 수행할 때마다 동적으로 연산 그래프를 생성한다.
따라서, 연산 결과물이 어떤 텐서로부터 어떤 연산을 통해 왔는지 추적할 수 있다.
그 결과, 우리는 최종 결과물인 스칼라에 역전파 알고리즘을 통해 미분을 수행하면 자식 노드에 해당하는
텐서와 연산을 자동으로 찾을 수 있다.
'''

# x,y(leaf) -> x+y (parent)
# x+y + (2,2)(leaf) -> z (root)
# z 부터 역전파를 수행하면, 이미 생성된 연산 그래프에 따라 미분 값을 전달한다.
z = (x+y) + torch.FloatTensor(2,2)
#print(z)

# 기울기를 구할 필요가 없는 연산의 경우, (비학습, 예측, 추론 등)
with torch.no_grad():
    z = (x+y) + torch.FloatTensor(2,2)

# 3.피드포워드, 선형계층, 완전연결계층, FC Layer 구현 =====================
def linear(x, W, b):
    y = torch.mm(x, W) + b
    return y

x = torch.FloatTensor(16,10)
W = torch.FloatTensor(10,5)
b = torch.FloatTensor(5)

y=linear(x,W,b)
#print(y.size())

# 4.nn.Module 클래스, 상속하여 사용자 정의 클래스를 만들 수 있다. ================
import torch.nn as nn
class MyLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyLinear, self).__init__()
        self.linear = nn.Linear(input_size,output_size)

    def forward(self, x):
        y = self.linear(x)
        return y

x = torch.FloatTensor(16,10) # 10개 원소를 갖는 16개의 벡터를 가진 행렬 x
linear = MyLinear(10,5)
y = linear(x) # x 를 MyLinear 클래스의 객체인 linear에 통과
#print(y.size()) # 결과 : 5개의 원소를 갖는 벡터로 변환 (16,10)->(16,5)
print(linear) # (linear): Linear(in_features=10, out_features=5, bias=True), 내부 가중치 파라미터 확인

# 5. 역전파 수행 ==================================
'''
연산을 통해 값을 앞으로 전달하는 피드포워드 구현
다음은, 피드포워드를 뒤로 보내는 역전파 구현할 차례.
피드포워드로 얻은 값에서 실제 정답값과의 차이를 계산하여 오류(손실)을 뒤로 전달하자.
에러 값은 꼭 스칼라여야 한다.
'''
objective = 100 # 원하는 값, 정답값

x = torch.FloatTensor(16,10)
linear = MyLinear(10,5)
y = linear(x)
loss = (objective - y.sum()**2) # linear의 결과값 텐서의 합과 목표값의 거리를 구한다.
#print(loss)
#loss.backword() # 거리값을 backward()함수를 사용해 기울기를 구한다.

# 6. train()과 eval() =====================================
# 학중...(디폴트 : 훈련 모드)
linear.eval()
# 추론모드 적용 : 드롭아웃, 배치 정규화 ...
linear.train()
# 다시 학습 재시작..습

# 7. GPU 사용하기, 텐서와 모델을 gpu 메모리로 보낸다.
'''
x = torch.cuda.FloatTensor(16,10) # 텐서를 gpu 메모리에 생성한다. (복사 혹은 이)
linear = MyLinear(10,5)
linear.cuda() # 모델도 gpu 메모리에 보낸다.
y = linear(x)
'''
