from numpy import True_
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

class MetaClassifier(nn.Module):
    def __init__(self):
        super(MetaClassifier, self).__init__()
        self.phi1 = nn.Sequential(
            nn.Linear(105, 1),
            nn.ELU(inplace = True)
        )
        self.phi2 = nn.Sequential(
            nn.Linear(65, 1),
            nn.ELU(inplace = True)
        )
        self.phi3 = nn.Sequential(
            nn.Linear(33, 1),
            nn.ELU(inplace = True)
        )
        self.phi4 = nn.Sequential(
            nn.Linear(17, 1),
            nn.ELU(inplace = True)
        )
        self.rho = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        n1 = self.phi1(x1)
        L1 = n1.sum(1)
        n1_out = n1

        n1 = n1.permute(0,2,1).expand(-1,16,-1)
        x2 = torch.cat((x2,n1),2)
        n2 = self.phi2(x2)
        L2 = n2.sum(1)
        n2_out = n2

        n2 = n2.permute(0,2,1).expand(-1,8,-1)
        x3 = torch.cat((x3,n2),2)
        n3 = self.phi3(x3)
        L3 = n3.sum(1)
        n3_out = n3

        n3 = n3.permute(0,2,1)
        x4 = torch.cat((x4,n3),2)
        n4 = self.phi4(x4)
        L4 = n4.sum(1)
        n4_out = n4

        concat = torch.cat((L1, L2, L3, L4), 1)
        return self.rho(concat), n1_out, n2_out, n3_out, n4_out
        
    
    
class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.phi1 = nn.Sequential(
            nn.Linear(105, 1),
            nn.ELU(inplace=True)
        )
        self.phi2 = nn.Sequential(
            nn.Linear(33, 1),
            nn.ELU(inplace=True)
        )
        self.phi3 = nn.Sequential(
            nn.Linear(17, 1),
            nn.ELU(inplace=True)
        )
        self.phi4 = nn.Sequential(
            nn.Linear(9, 1),
            nn.ELU(inplace=True)
        )
        self.rho = nn.Sequential(
            nn.Linear(57, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        n1 = self.phi1(x1)
        n2 = self.phi2(x2)
        n3 = self.phi3(x3)
        n4 = self.phi4(x4)
        concat = torch.cat((n1, n2, n3, n4),1)
        concat = concat.permute(0,2,1)
        n1_out = n1
        n2_out = n2
        n3_out = n3
        n4_out = n4

        return self.rho(concat), n1_out, n2_out, n3_out, n4_out
    
    
class MetaWithVar(nn.Module):
    def __init__(self):
        super(MetaWithVar, self).__init__()
        self.phi1 = nn.Sequential(
            nn.Linear(105, 1),
            nn.ELU(inplace = True)
        )
        self.phi2 = nn.Sequential(
            nn.Linear(65, 1),
            nn.ELU(inplace = True)
        )
        self.phi3 = nn.Sequential(
            nn.Linear(33, 1),
            nn.ELU(inplace = True)
        )
        self.phi4 = nn.Sequential(
            nn.Linear(17, 1),
            nn.ELU(inplace = True)
        )
        self.rho = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        n1 = self.phi1(x1)
        s1 = n1.sum(1)
        ##Here torch.median returns a torch.return_types.median, includes indices and values
        i, m1 = n1.median(1)
        v1 = torch.var(n1,1)
        n1_out = n1

        n1 = n1.permute(0,2,1).expand(-1,16,-1)
        x2 = torch.cat((x2,n1),2)
        n2 = self.phi2(x2)
        s2 = n2.sum(1)
        i, m2 = n2.median(1)
        v2 = torch.var(n2,1)
        n2_out = n2


        n2 = n2.permute(0,2,1).expand(-1,8,-1)
        x3 = torch.cat((x3,n2),2)
        n3 = self.phi3(x3)
        s3 = n3.sum(1)
        i, m3 = n3.median(1)
        v3 = torch.var(n3, 1)
        n3_out = n3

        n3 = n3.permute(0,2,1)
        x4 = torch.cat((x4,n3),2)
        n4 = self.phi4(x4)
        s4 = n4.sum(1)
        i, m4 = n4.median(1)
        v4 = n4.squeeze(1)
        n4_out = n4

        var = torch.cat((s1, s2, s3, s4, v1, v2, v3, v4), 1)
        m = torch.cat((s1, s2, s3, s4, m1, m2, m3, m4), 1)
        v = torch.cat((v1, v2, v3, v4), 1)
        return self.rho(var), n1_out, n2_out, n3_out, n4_out
        
class MetaWithAcc(nn.Module):
    def __init__(self):
        super(MetaWithAcc, self).__init__()
        self.phi1 = nn.Sequential(
            nn.Linear(105, 1),
            nn.ELU(inplace = True)
        )
        self.phi2 = nn.Sequential(
            nn.Linear(65, 1),
            nn.ELU(inplace = True)
        )
        self.phi3 = nn.Sequential(
            nn.Linear(33, 1),
            nn.ELU(inplace = True)
        )
        self.phi4 = nn.Sequential(
            nn.Linear(17, 1),
            nn.ELU(inplace = True)
        )
        self.rho = nn.Sequential(
            nn.Linear(6, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4, a1, a2):
        n1 = self.phi1(x1)
        L1 = n1.sum(1)
        n1_out = n1

        n1 = n1.permute(0,2,1).expand(-1,16,-1)
        x2 = torch.cat((x2,n1),2)
        n2 = self.phi2(x2)
        L2 = n2.sum(1)
        n2_out = n2

        n2 = n2.permute(0,2,1).expand(-1,8,-1)
        x3 = torch.cat((x3,n2),2)
        n3 = self.phi3(x3)
        L3 = n3.sum(1)
        n3_out = n3

        n3 = n3.permute(0,2,1)
        x4 = torch.cat((x4,n3),2)
        n4 = self.phi4(x4)
        L4 = n4.sum(1)
        n4_out = n4

        a1 = a1.float()
        a2 = a2.float()
        concat = torch.cat((L1, L2, L3, L4, a1, a2), 1)
        return self.rho(concat), n1_out, n2_out, n3_out, n4_out



class MetaClassifier4Layers(nn.Module):
    def __init__(self):
        super(MetaClassifier4Layers, self).__init__()
        self.phi1 = nn.Sequential(
            nn.Linear(105, 105),
            nn.ELU(inplace=True),
            nn.Linear(105, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 1)
        )
        self.phi2 = nn.Sequential(
            nn.Linear(65, 65),
            nn.ELU(inplace=True),
            nn.Linear(65, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 1),
        )
        self.phi3 = nn.Sequential(
            nn.Linear(33, 33),
            nn.ELU(inplace=True),
            nn.Linear(33, 1),
        )
        self.phi4 = nn.Sequential(
            nn.Linear(17, 17),
            nn.ELU(inplace=True),
            nn.Linear(17, 1),
        )
        self.rho = nn.Sequential(
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        n1 = self.phi1(x1)
        L1 = n1.sum(1)

        n1 = n1.permute(0,2,1).expand(-1,16,-1)
        x2 = torch.cat((x2,n1),2)
        n2 = self.phi2(x2)
        L2 = n2.sum(1)

        n2 = n2.permute(0,2,1).expand(-1,8,-1)
        x3 = torch.cat((x3,n2),2)
        n3 = self.phi3(x3)
        L3 = n3.sum(1)

        n3 = n3.permute(0,2,1).expand(-1,2,-1)
        x4 = torch.cat((x4,n3),2)
        n4 = self.phi4(x4)
        L4 = n4.sum(1)

        concat = torch.cat((L1, L2, L3, L4), 1)
        return self.rho(concat)






class DirectConnect(nn.Module):
    def __init__(self):
        super(DirectConnect, self).__init__()
        self.phi1 = nn.Sequential(
            nn.Linear(105, 1),
            nn.ELU(inplace=True)
        )
        self.phi2 = nn.Sequential(
            nn.Linear(33, 1),
            nn.ELU(inplace=True)
        )
        self.phi3 = nn.Sequential(
            nn.Linear(17, 1),
            nn.ELU(inplace=True)
        )
        self.phi4 = nn.Sequential(
            nn.Linear(9, 1),
            nn.ELU(inplace=True)
        )
        self.rho = nn.Sequential(
            nn.Linear(4, 1),
            nn.ELU(inplace=True)
        )

    def forward(self, x1, x2, x3, x4):
        n1 = self.phi1(x1)
        L1 = n1.sum(dim=1)

        n2 = self.phi2(x2)
        L2 = n2.sum(dim=1)

        n3 = self.phi3(x3)
        L3 = n3.sum(dim=1)

        n4 = self.phi4(x4)
        L4 = n4.sum(dim=1)

        concat = torch.cat((L1, L2, L3, L4), 1)
        return self.rho(concat)
