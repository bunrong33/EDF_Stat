import torch 

path = "../data/conso_france_2022/blocks_missing_1.pt"
test = torch.load(path)
target = test[:,0,:]

print(type(test))
print(test.shape)
print(test.dtype)
# print(test)
print("missing values:", torch.isnan(target).sum())
print("total missing:", torch.isnan(target).sum().item())
print("missing per series:", torch.isnan(target).sum(dim=1))
print("unique missing counts:", torch.unique(torch.isnan(target).sum(dim=1)))