import torch
import torch.nn as nn
import torch.nn.functional as F

class HOE_model(nn.Module):
    def __init__(self, nfeat, nclass, dropout_rate=0.8, k_size=5):
        super(HOE_model, self).__init__()

        self.fc0 = nn.Linear(nfeat, 512)

        self.conv = nn.Conv1d(in_channels=512,out_channels=512,kernel_size=k_size)

        self.fc1_1 = nn.Linear(512, 128)
        self.fc1_2 = nn.Linear(128, 1)

        
        self.fc2_1 = nn.Linear(512, 128)
        self.fc2_2 = nn.Linear(128, 128)

        self.dropout_rate = dropout_rate
        self.k_size = k_size
        # if self.training:
        #     self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.fill_(1)
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, input):

        fea = F.relu(self.fc0(input))
        fea = F.dropout(fea, self.dropout_rate, training=self.training)
        # temporal
        old_fea = fea.permute(0,2,1) 
        old_fea = torch.nn.functional.pad(old_fea, (self.k_size//2,self.k_size//2), mode='replicate')
        new_fea = self.conv(old_fea)
        new_fea = new_fea.permute(0,2,1)

        # semantic
        x = F.relu(self.fc1_1(new_fea))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = torch.sigmoid(self.fc1_2(x))
        # dynamic
        new_fea = F.relu(self.fc2_1(new_fea))
        new_fea = F.dropout(new_fea, self.dropout_rate, training=self.training)
        new_fea = self.fc2_2(new_fea)

        return x, new_fea

    def val(self, input):
        # assert (x.shape[0] == 1)
        # original layers
        # import pdb;pdb.set_trace()
        input_new = input.unsqueeze(0)
        at_1 = self.__self_module__(input_new,1)#1,1,1,32,1
        input_view = input.view((1,1,1,32,-1))
        x_1 = torch.sum(input_view*at_1,-2)#1,1,1,1024
        
        x =  torch.cat([input_new,x_1,],2)
        x = self.__Score_pred__(x)

        return x, at_1, at_1

class Self_att(nn.Module):
    def __init__(self, nfeat, nclass, is_train, dropout_rate=0.6):
        super(Self_att, self).__init__()
        # original layers

        self.at1 = nn.Linear(nfeat, 256)
        self.at2 = nn.Linear(256, 64)
        self.at3 = nn.Linear(64, nclass)
        # self.training = is_train
        self.dropout_rate = dropout_rate


    def forward(self, input, in_num):
        # assert (x.shape[0] == 1)
        # original layers
        at = F.relu(self.at1(input))
        at = F.dropout(at, self.dropout_rate, training=self.training)
        at = F.dropout(self.at2(at), self.dropout_rate, training=self.training)
        at = self.at3(at)
        # import pdb;pdb.set_trace()
        at = at.view((input.shape[0],input.shape[1],in_num,-1,1))
        # import pdb;pdb.set_trace()
        # at = F.relu(at)
        at = torch.sigmoid(at)
        
        
        return at

class Score_pred(nn.Module):
    def __init__(self, nfeat, nclass, is_train, dropout_rate=0.6):
        super(Score_pred, self).__init__()
        # original layers
        self.fc1 = nn.Linear(nfeat, 512)
        self.fc2 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, nclass)
        # self.training = is_train
        self.dropout_rate = dropout_rate

    def forward(self, input):
        # assert (x.shape[0] == 1)
        # original layers

        x = F.relu(self.fc1(input))
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc3(x)
        x = F.sigmoid(x)

        return x