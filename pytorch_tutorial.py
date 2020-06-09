import torch
import numpy as np

x = torch.Tensor([[1,2],[3,4]])
x = torch.from_numpy(np.array([[1,2],[3,4]]))
x = np.array([[1,2],[3,4]])
print(x)
