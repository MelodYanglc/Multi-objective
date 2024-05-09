import pandas as pd
import torch
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# from EarlyStoppingUtil import EarlyStopping
from net.DynamicConv import *
from sklearn.metrics import recall_score

import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=20, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # path = os.path.join(self.save_path, 'best_network.pth')
        path = self.save_path
        # torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        torch.save(model, path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

class MyData(Dataset):
    def __init__(self,dataX,dataY):
        self.dataX = np.array(dataX)
        self.dataY = np.array(dataY)

    def __len__(self):
        return len(self.dataY)

    def __getitem__(self, index):
        dataX = self.dataX[index,:]
        dataY = self.dataY[index]
        sample = {'dataX':dataX,'dataY':dataY}
        return sample


class MLP(nn.Module):
    def __init__(self, input_size, common_size):
        super(MLP, self).__init__()
        hidden_num = int(input_size * 2)
        self.inputLayer = nn.Sequential(
            nn.Linear(input_size,  hidden_num),
            # nn.BatchNorm1d(hidden_num, eps=0.001, momentum=0.03),
            nn.ReLU(inplace=True),
        )
        self.hiddenLayer = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(hidden_num, hidden_num),
            # nn.BatchNorm1d(hidden_num, eps=0.001, momentum=0.03),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.Linear(hidden_num, hidden_num),
            # nn.BatchNorm1d(hidden_num, eps=0.001, momentum=0.03),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
        )
        self.outputLayer = nn.Sequential(
            nn.Linear(hidden_num, common_size),
        )

    def forward(self, x):
        print("x:",x.shape)
        x = self.inputLayer(x)
        print("x:",x.shape)
        x = self.hiddenLayer(x)
        out = self.outputLayer(x)
        return out

class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)#nn.LayerNorm(out_channels,eps=1e-6)#
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # squeeze操作
        y = self.fc(y).view(b, c, 1, 1)  # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x)  # 注意力作用每一个通道上


class selfattention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.key = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        batch_size, channels, height, width = input.shape
        # input: B, C, H, W -> q: B, H * W, C // 8
        q = self.query(input).view(batch_size, -1, height * width).permute(0, 2, 1)
        # input: B, C, H, W -> k: B, C // 8, H * W
        k = self.key(input).view(batch_size, -1, height * width)
        # input: B, C, H, W -> v: B, C, H * W
        v = self.value(input).view(batch_size, -1, height * width)
        # q: B, H * W, C // 8 x k: B, C // 8, H * W -> attn_matrix: B, H * W, H * W
        attn_matrix = torch.bmm(q, k)  # torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)  # 经过一个softmax进行缩放权重大小.
        out = torch.bmm(v, attn_matrix.permute(0, 2, 1))  # tensor.permute将矩阵的指定维进行换位.这里将1于2进行换位。
        out = out.view(*input.shape)

        return self.gamma * out + input

class SpatialAttentionModule(nn.Module):
    def __init__(self,inchannel):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d =nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1
                      , padding=3
            )
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 16,1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // 16, in_planes,1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class GCNet(nn.Module):
    def __init__(self, inplanes, planes=16, pool='att', fusions=['channel_add'], ratio=8):
        super(GCNet, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)#context Modeling
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes // ratio, kernel_size=1),
                nn.LayerNorm([self.planes // ratio, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes // ratio, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)#softmax操作
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class MLPCNN(nn.Module):
    def __init__(self, inchannels, outChannels,reshape=(6,9)):
        super(MLPCNN, self).__init__()
        self.reshape = reshape

        self.features = self.conv_block(inchannels,16,3)
        self.encoder1 = self.conv_block(16, 16,3)
        self.encoder2 = self.conv_block(32, 16,3)
        self.encoder3 = self.conv_block(48, 16,3)
        self.attention1 = GCNet(16)
        self.attention2 = GCNet(16)
        self.attention3 = GCNet(16)

        self.classifyer = nn.Sequential(
            # nn.Linear(self.reshape[0] * self.reshape[1] * 8, self.reshape[0] * self.reshape[1] * 4),
            # nn.ReLU(inplace=True),
            nn.Linear(self.reshape[0] * self.reshape[1] * 64, outChannels),
            # nn.Dropout(0.3),
            # nn.Linear(self.reshape[0] * self.reshape[1] * 24, outChannels)
        )

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3):
        """
        The conv block of common setting: conv -> relu -> bn
        In conv operation, the padding = 1
        :param in_channels: int, the input channels of feature
        :param out_channels: int, the output channels of feature
        :param kernel_size: int, the kernel size of feature
        :return:
        """
        block = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            nn.ReLU(inplace=True),
        )
        return block

    @staticmethod
    def concat(f1, f2):
        """
        Concat two feature in channel direction
        """
        return torch.cat((f1, f2), 1)

    def forward(self, x):
        # print("x shape:",x.shape)
        (b,l) = x.shape
        # inputX = torch.reshape(x,(b,1,5,5))
        inputX = torch.reshape(x, (b, 1, self.reshape[0], self.reshape[1]))

        features = self.features(inputX)  # 16
        # encoder
        encoder1 = self.encoder1(features)  # 16
        attention1 = self.attention1(features)  # 16
        output1 = torch.cat((attention1, encoder1), 1)  # 16 + 16 = 32
        encoder2 = self.encoder2(output1)  # 16
        attention2 = self.attention2(encoder1)  # 16
        output2 = torch.cat((attention1, attention2, encoder2), 1)  # 16 + 16 + 16 = 48
        # output2 = torch.cat((attention1,attention2, encoder2), 1)  # 16 + 16 + 16 = 48
        encoder3 = self.encoder3(output2)  # 16
        attention3 = self.attention3(encoder2)  # 16
        output3 = torch.cat((attention1, attention2, attention3, encoder3), 1)  # 16 + 16 + 16 + 16 = 64
        # output3 = torch.cat((attention1,attention2,attention3, encoder3), 1)  # 16 + 16 + 16 + 16 = 64
        # print("output3:", output3.shape)
        out = output3.view(b,-1)
        out = self.classifyer(out)
        # print("out:",out.shape)
        return out

class CNNResNet(nn.Module):
    def __init__(self, inchannels, outChannels,reshape=(6,9)):
        super(CNNResNet, self).__init__()
        self.reshape = reshape

        self.conv1 = self.conv_block(inchannels,4,3)
        self.conv2 = self.conv_block(4,4,3)
        self.conv3 = self.conv_block(4,4,3)

        self.classifyer = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(self.reshape[0] * self.reshape[1] * 4, outChannels)
        )

    def forward(self,x):
        (b, l) = x.shape
        inputX = torch.reshape(x, (b, 1, self.reshape[0], self.reshape[1]))
        x = self.conv1(inputX)
        x = self.conv2(x)
        x = self.conv3(x)
        out = x.view(b, -1)
        out = self.classifyer(out)
        return out

    @staticmethod
    def conv_block(in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03),
            nn.ReLU(inplace=True),
        )
        return block


from einops.layers.torch import Rearrange
class FeedForward(nn.Module):
    def __init__(self,dim,hidden_dim,dropout=0.):
        super().__init__()
        self.net=nn.Sequential(
            #由此可以看出 FeedForward 的输入和输出维度是一致的
            nn.Linear(dim,hidden_dim),
            #激活函数
            nn.GELU(),
            #防止过拟合
            nn.Dropout(dropout),
            #重复上述过程
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout)
            # nn.Linear(dim, dim),
            # nn.ReLU(inplace=True),
        )
    def forward(self,x):
        x=self.net(x)
        return x

class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')

        )
        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self, in_channels, dim, num_classes, patch_size, image_size, depth, token_dim, channel_dim,
                 dropout=0.,reshape=(6,9)):
        super().__init__()
        self.reshape = reshape
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2  # （224/16）**2=196
        # embedding 操作，看见没用卷积来分成一小块一小块的
        # 通过embedding可以将这张3*224*224的图片转换为Channel*Patches=512*196，再通过Rearrange转为196*512
        self.to_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
            )
        num_patches = (image_size // patch_size) * (image_size // patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # 输入为196*512的table
        # 以下为token-mixing MLPs（MLP1）和channel-mixing MLPs（MLP2）各一层
        self.mixer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(dim, self.num_patches, token_dim, channel_dim, dropout))

        self.layer_normal = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        (b, l) = x.shape
        x = torch.reshape(x, (b, 1, self.reshape[0], self.reshape[1]))
        x = self.to_embedding(x)
        # b1, n1, _ = x.shape  # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值
        # x += self.pos_embedding[:, :(n1)]
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x = self.layer_normal(x)
        x = x.mean(dim=1)

        x = self.mlp_head(x)
        return x

def trainModel1(criterion,net,optimizer,epochs,train_loader,test_loader,scheduler,saveLossFigName,device):
    train_acc_all= []
    test_acc_all = []
    # #早停法，过拟合解决策略
    # best_model_path = 'modelCNN\\bestModel\\' + saveLossFigName
    # early_stopping = EarlyStopping(best_model_path,patience=50)
    for epoch in range(epochs):
        #训练
        epcoh_train_correct,yTrainTure,yTrainPred = train(criterion, net, optimizer, epochs, train_loader, scheduler,device)
        #测试
        epcoh_test_correct,yTestTure,yTestPred = test(criterion, net, test_loader,device)
        train_acc_all.append(float(epcoh_train_correct))
        test_acc_all.append(float(epcoh_test_correct))
        scheduler.step()
        # # 早停止
        # early_stopping(epcoh_test_correct, net)
        # # 达到早停止条件时，early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break  # 跳出迭代，结束训练
        if epoch == (epochs - 1):
            recaltrain = recall_score(yTrainTure,yTrainPred)
            recaltest = recall_score(yTestTure,yTestPred)
            print("recaltrain:",recaltrain," recaltest:",recaltest)
    # 保存loss曲线
    save_loss_figures(train_acc_all, test_acc_all,saveLossFigName)
    return train_acc_all,test_acc_all,train_acc_all[-1],test_acc_all[-1]

def train(criterion,net,optimizer,epochs,train_loader,scheduler,device):
    finnal_correct = 0
    net.train()
    epoch_loss = 0
    epoch_total = 0
    epoch_correct = 0
    yTrue = []
    yPred= []
    for i, data in enumerate(train_loader):
        x, y = torch.tensor(data['dataX']).float(), torch.tensor(data['dataY'])
        x,y= x.to(device),y.to(device)
        optimizer.zero_grad()
        output = net(x)
        # loss = criterion(net, output, y)
        loss = criterion(output,y)
        # regular = 0
        # for param in net.parameters():
        #     regular += torch.sum(torch.abs(param))
        # loss += 1 * regular
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        _, predicted = torch.max(output.data, 1)
        epoch_total += y.size(0)
        epoch_correct += (predicted == y).sum()

        yTrue.append(y.cpu().detach().numpy().tolist())
        yPred.append(predicted.cpu().detach().numpy().tolist())

    finnal_correct = 100.0 * (epoch_correct / epoch_total)
    # if epoch % 20 == 0:
    #     print("Train loss total:",(epoch_loss / epoch_total))
    # if epoch == (epochs - 1):
    #     finnal_correct = 100.0 * (epoch_correct / epoch_total)

    # print("Accuracy on the train set: %.2d %%" % finnal_correct)
    # print("yTrue:",yTrue)
    # print("yPred:", yPred)

    return finnal_correct,sum(yTrue,[]),sum(yPred,[])

def test(criterion,net,test_loader,device):
    loss_total = 0
    total = 0
    correct = 0
    finnal_correct = 0
    net.eval()
    # predictedAll = []
    yTrue = []
    yPred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            x, y = torch.tensor(data['dataX']).float(), torch.tensor(data['dataY'])
            x,y= x.to(device),y.to(device)

            output = net(x)
            # loss = criterion(net, output, y)
            loss = criterion(output,y)
            loss_total += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum()
            # predictedAll.append(predicted)

            yTrue.append(y.cpu().detach().numpy().tolist())
            yPred.append(predicted.cpu().detach().numpy().tolist())

    finnal_correct = (100.0 * correct / total)
    # print(predictedAll)
    # print("acc:",accuracy_score(predictedAll,data['dataY']))
    # print("Accuracy on the test set: %.2d %%" % finnal_correct)
    # print("Test loss total:", (loss_total / total))
    # print(sum(yTrue,[]))
    return finnal_correct,sum(yTrue,[]),sum(yPred,[])

def trainModel2(criterionTrain,criterionTest,net,optimizer,epochs,train_loader,test_loader,scheduler,saveLossFigName,device):
    train_loss_all= []
    test_loss_all = []
    # #早停法，过拟合解决策略
    # best_model_path = 'modelCNN\\bestModel\\' + saveLossFigName
    # early_stopping = EarlyStopping(best_model_path,patience=50)
    for epoch in range(epochs):
        #训练
        epcoh_train_correct = train2(criterionTrain,net,optimizer,epochs,train_loader,scheduler,device)
        #测试
        epcoh_test_correct = test2(criterionTest, net, test_loader,device)
        train_loss_all.append(float(epcoh_train_correct))
        test_loss_all.append(float(epcoh_test_correct))
        scheduler.step()
        # print(scheduler.get_last_lr()[0])
        # # 早停止
        # early_stopping(epcoh_test_correct, net)
        # # 达到早停止条件时，early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break  # 跳出迭代，结束训练
    #保存loss曲线
    save_loss_figures(train_loss_all, test_loss_all,saveLossFigName)
    return train_loss_all,test_loss_all,train_loss_all[-1],test_loss_all[-1]

def train2(criterion,net,optimizer,epochs,train_loader,scheduler,device):
    lossFinal = 0
    loss_total = 0
    total_num = 0
    for i, data in enumerate(train_loader):
        x, y = torch.tensor(data['dataX']).float(), torch.tensor(data['dataY']).float()
        # print(x.shape)
        x,y= x.to(device),y.to(device)
        # print("x:", x.shape)
        optimizer.zero_grad()
        output = net(x)
        # loss = criterion(net,output, y)
        # print("output:",output.shape)
        loss = criterion(output,y)
        # regular = 0
        # for param in net.parameters():
        #     regular += torch.sum(torch.abs(param))
        # loss += 0.001 * regular
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
        total_num += 1
        # if epoch == (epochs-1):
        #     lossFinal = loss_total / total_num
    lossFinal = loss_total / total_num
    # if epoch % 20 == 0:
    #     print("Train loss total: {}".format(loss_total / total_num))
    # print("Train mse total: {}".format(lossFinal))
    return lossFinal

def test2(criterion,net,test_loader,device):
    from sklearn.metrics import mean_squared_error
    loss_total = 0
    total_num = 0
    net.eval()
    with torch.no_grad():
        test_loss = []
        for i, data in enumerate(test_loader):
            x, y = torch.tensor(data['dataX']).float(), torch.tensor(data['dataY']).float()
            # print(x.shape)
            x,y= x.to(device),y.to(device)

            output = net(x)
            # loss = criterion(net, output, y)
            loss = criterion(output,y)
            loss_total += loss.item()
            # print("output:",output.squeeze(1).cpu().shape," y:",y.cpu().shape)
            total_num += 1

    # print("Test mse total:{}".format(loss_total / total_num))
    return loss_total / total_num

def save_loss_figures(tran_loss,test_loss,saveLossFigName):
    plt.clf()
    counter = range(len(tran_loss))
    plt.plot(counter,tran_loss)
    plt.plot(counter,test_loss)
    plt.legend(['train loss/acc','test loss/acc'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(saveLossFigName + ".png")

if __name__ == '__main__':
    device = "cuda"
    model = MLPMixer(in_channels=1, dim=12, num_classes=1, patch_size=1, image_size=7, depth=1,
                     token_dim=12,
                     channel_dim=12).to(device)
    img = torch.randn(4, 1, 7, 7).to(device).float()
    print(img.shape)
    output = model(img)
    print(output.shape)


