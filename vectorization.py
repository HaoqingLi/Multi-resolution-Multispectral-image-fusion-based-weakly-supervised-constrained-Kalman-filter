import torch
import numpy as np
def vectorization(x):
    if len(x.size()) == 2:
        x = torch.unsqueeze(x, 2)
    y = torch.zeros(x.size(0) * x.size(1) * x.size(2), 1)
    ind2 = torch.linspace(x.size(1) - 1, 0, x.size(1)).numpy()
    for i in range(x.size(0)):
        if np.mod(i+1, 2) == 0:
            for j in range(x.size(1)):
                if np.mod(ind2[j] + i + 1, 2) == 0:
                    ind1 = torch.linspace(i * x.size(1) * x.size(2) + j * x.size(2), i* x.size(1) * x.size(2) + j * x.size(2) + x.size(2)-1, x.size(2)).numpy()
                    y[ind1, 0] = torch.flip(x[i, int(ind2[j]), :], [0])
                else:
                    ind1 = torch.linspace(i * x.size(1) * x.size(2) + j * x.size(2), i* x.size(1) * x.size(2) + j * x.size(2) + x.size(2)-1, x.size(2)).numpy()
                    y[ind1, 0] = x[i, int(ind2[j]), :]
        else:
            for j in range(x.size(1)):
                if np.mod(j + i + 1, 2) == 0:
                    ind1 = torch.linspace(i * x.size(1) * x.size(2) + j * x.size(2),
                                          i * x.size(1) * x.size(2) + j * x.size(2) + x.size(2) - 1, x.size(2)).numpy()
                    y[ind1, 0] = torch.flip(x[i, j, :], [0])
                else:
                    ind1 = torch.linspace(i * x.size(1) * x.size(2) + j * x.size(2),
                                          i * x.size(1) * x.size(2) + j * x.size(2) + x.size(2) - 1, x.size(2)).numpy()
                    y[ind1, 0] = x[i, j, :]

    return y



# def matlabreshape(x, dim):
#     y = torch.zeros(dim)
#     if len(dim) == 3:
#         windowsize = int(x.size(0)/dim[2])
#         for i in range(dim[2]):
#             y[:, :, i] = torch.reshape(x[i*windowsize:(i+1)*windowsize], [dim[1], dim[0]]).T
#     else:
#         y = torch.reshape(x, [dim[1], dim[0]]).T
#     return y

# N = 16
# x = torch.reshape(torch.arange(N).view(N, 1).float(), [4,2,2])
# y = vectorization(x)
# print(x.numpy()+1)
# print(y.numpy()+1)