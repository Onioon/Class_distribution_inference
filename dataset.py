import torch
import numpy as np
from target import net
from target import accuracy

class NetParasDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def __len__(self):
        return 2000

    def __getitem__(self, index):
        # generates one sample of data
        l1, l2, l3, l4 = torch.load('meta_train/' + str(index) +'.pt')
        label = index%100/100
        sample = {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "Label": label}
        return sample

class para_n_acc(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        # generates one sample of data
        l1, l2, l3, l4, h, l = torch.load('data_acc/' + str(index) + '.pt')
        label = index % 100 / 100
        sample = {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "Label": label, "h":h, "l":l}
        return sample
        

# class paras_with_acc(torch.utils.data.Dataset):
#     def __init__(self, high_loader, low_loader):
#         self.high_loader = high_loader
#         self.low_loader = low_loader
#
#     def __len__(self):
#         return 2000
#
#     def __getitem__(self, index):
#         # generates one sample of data
#         l1, l2, l3, l4 = torch.load('meta_train/' + str(index) +'.pt')
#         label = index%100/100
#         model = net()
#         paras = {"fc1.weight": l1[:,:104], "fc1.bias": l1[:,104],
#                 "fc2.weight": l2[:,:32], "fc2.bias": l2[:,32],
#                 "fc3.weight": l3[:,:16], "fc3.bias": l3[:,16],
#                 "fc4.weight": l4[:,:8], "fc4.bias": l4[:,8]}
#         model.load_state_dict(paras)
#         h_test, l_test = self.high_loader, self.low_loader
#         h_acc = accuracy(model, h_test)
#         l_acc = accuracy(model, l_test)
#         sample = {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "h_acc": np.float(h_acc), "l_acc": np.float(l_acc), "Label": label}
#         return sample
