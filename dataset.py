import torch

class NetParasDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def __len__(self):
        return 1000

    def __getitem__(self, index):
        # generates one sample of data
        l1, l2, l3, l4 = torch.load('meta_train/' + str(index) +'.pt')
        label = index%100/100
        sample = {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "Label": label}
        return sample

