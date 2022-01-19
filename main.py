import torch
import torch.nn as nn
from MY_MODELS import Datacon1model
from torch.optim import AdamW, Adam
from torch.nn import CrossEntropyLoss

class Datacon1classifier(nn.Module):
    def __init__(self,modelKind,backboneOutFeature, cropLinNum,disriskLinNum,
                 totalCropNum,totalDisriskNum,lr=3e-4,eps=1e-9):
        super(Datacon1classifier,self).__init__()

        self.modelKind = modelKind
        self.backboneOutFeature = backboneOutFeature
        self.cropLinNum = cropLinNum
        self.disriskLinNum = disriskLinNum
        self.totalCropNum = totalCropNum
        self.totalDisriskNum = totalDisriskNum

        self.lr = lr
        self.eps = eps

        ###################MODEL SETTING###########################

        self.Datacon1Model = Datacon1model(
            modelKind=self.modelKind,
            backboneOutFeature=backboneOutFeature,
            cropLinNum=cropLinNum,
            disriskLinNum=disriskLinNum,
            totalCropNum=totalCropNum,
            totalDisriskNum=totalDisriskNum
        )

        self.loss_lst_trn = []
        self.loss_lst_val = []

        self.acc_lst_trn = []
        self.acc_lst_val = []

        self.optimizer = Adam(self.Datacon1Model.parameters(),
                              lr=self.lr,  # 학습률
                              eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )

    def forward(self,x):

        out1,out2 = self.Datacon1Model(x)

        return out1,out2

    def calLossCrop(self,logit,label):

        loss = CrossEntropyLoss()

        return loss(logit,label)

    def calLossDisrisk(self,logit,label):

        loss = CrossEntropyLoss()

        return loss(logit,label)

    def trainingStep(self):












