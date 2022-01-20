import torch
import torch.nn as nn
from MY_MODELS import Datacon1model
from torch.optim import AdamW, Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from DataLoading import MyDatacon1Dataset
from save_funcs import createDirectory,mk_name
import numpy as np
import matplotlib.pyplot as plt
import time
import csv


class Datacon1classifier(nn.Module):
    def __init__(self,modelKind,backboneOutFeature, LinNum,
                 totalCropNum,data_folder_dir_trn,
                 data_folder_dir_val,MaxEpoch,data_folder_dir_test,
                 modelPlotSaveDir,iter_to_accumul,MaxStep,MaxStepVal,
                 bSizeTrn= 32,bSizeVal=10,lr=3e-4,eps=1e-9):


        super(Datacon1classifier,self).__init__()

        self.data_folder_dir_trn = data_folder_dir_trn
        self.data_folder_dir_val = data_folder_dir_val
        self.data_folder_dir_test = data_folder_dir_test

        self.modelKind = modelKind
        self.backboneOutFeature = backboneOutFeature
        self.LinNum = LinNum
        self.totalCropNum = totalCropNum


        self.iter_to_accumul = iter_to_accumul
        self.MaxStep = MaxStep
        self.MaxStepVal = MaxStepVal


        self.lr = lr
        self.eps = eps
        self.bSizeTrn = bSizeTrn
        self.bSizeVal = bSizeVal

        self.modelPlotSaveDir = modelPlotSaveDir

        ###################MODEL SETTING###########################
        print('failed loading model, loaded fresh model')
        self.Datacon1Model = Datacon1model(
            modelKind=self.modelKind,
            backboneOutFeature=backboneOutFeature,
            LinNum=LinNum,
            totalCropNum=totalCropNum,
            )

        USE_CUDA = torch.cuda.is_available()
        print(USE_CUDA)
        self.device = torch.device('cuda' if USE_CUDA else 'cpu')
        print('학습을 진행하는 기기:', self.device)

        self.loss_lst_trn = []
        self.loss_lst_trn_tmp = []
        self.loss_lst_val = []
        self.loss_lst_val_tmp = []

        self.acc_lst_trn = []
        self.acc_lst_trn_tmp = []

        self.acc_lst_val = []
        self.acc_lst_val_tmp = []

        self.num4epoch = 0
        self.MaxEpoch = MaxEpoch

        self.optimizer = Adam(self.Datacon1Model.parameters(),
                              lr=self.lr,  # 학습률
                              eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )

        MyTrnDataset = MyDatacon1Dataset(data_folder_dir=self.data_folder_dir_trn,TRAIN=True)
        self.trainDataloader = DataLoader(MyTrnDataset,batch_size=self.bSizeTrn,shuffle=True)

        MyValDataset = MyDatacon1Dataset(data_folder_dir=self.data_folder_dir_val, TRAIN=True)
        self.valDataloader = DataLoader(MyValDataset, batch_size=self.bSizeVal, shuffle=False)

        MyTestDataset = MyDatacon1Dataset(data_folder_dir=self.data_folder_dir_test,TRAIN=False)
        self.testLen = len(MyTestDataset)
        self.TestDataloader = DataLoader(MyTestDataset,batch_size=1,shuffle=False)

        self.Datacon1Model.to(device=self.device)



    def forward(self,x):

        out = self.Datacon1Model(x)

        return out

    def calLoss(self,logit,label):

        loss = CrossEntropyLoss()

        pred = torch.argmax(logit,dim=1)


        acc = torch.mean((pred == label).float())


        return loss(logit,label) , acc


    def trainingStep(self,trainingNum):

        self.Datacon1Model.train()
        countNum = 0

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            globalTime= time.time()

            for _,bInput, bLabel  in self.trainDataloader:
                localTime= time.time()

                bInput = bInput.to(self.device)

                bLogit = self.forward(bInput)
                bLogit = bLogit.cpu()


                ResultLoss,Acc = self.calLoss(bLogit,bLabel.long())

                ResultLoss.backward()
                self.loss_lst_trn_tmp.append(float(ResultLoss.item()))
                self.acc_lst_trn_tmp.append(Acc)

                if (countNum + 1) % self.iter_to_accumul == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if (countNum + 1 ) % self.bSizeVal == 0:
                    ################# mean of each append to lst for plot###########
                    self.loss_lst_trn.append(np.mean(self.loss_lst_trn_tmp))
                    self.acc_lst_trn.append(np.mean(self.acc_lst_trn_tmp))

                    ################# mean of each append to lst for plot###########
                    ###########flush#############
                    self.loss_lst_trn_tmp = []
                    self.acc_lst_trn_tmp = []

                if countNum == self.MaxStep:
                    break
                else:
                    countNum += 1

                localTimeElaps = round(time.time() - localTime,2)
                globalTimeElaps = round(time.time() - globalTime,2)

                print(f'globaly {globalTimeElaps} elapsed and locally {localTimeElaps} elapsed for {countNum} / {self.MaxStep}')
                print(f'num4epoch is : {self.num4epoch} and self.max_epoch : {self.MaxEpoch}')



        torch.set_grad_enabled(False)
        self.Datacon1Model.eval()

    def valdatingStep(self,validatingNum):

        self.Datacon1Model.eval()
        countNum = 0
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            for _,valBInput, valBLabel in self.valDataloader:

                valBInput = valBInput.to(self.device)

                valBLogit = self.forward(valBInput)
                valBLogit = valBLogit.cpu()

                ResultLoss,Acc = self.calLoss(valBLogit,valBLabel.long())

                self.loss_lst_val_tmp.append(float(ResultLoss.item()))
                self.acc_lst_val_tmp.append(Acc)

                if countNum == self.MaxStepVal:
                    break
                else:
                    countNum += 1

            self.loss_lst_val.append(np.mean(self.loss_lst_val_tmp))
            self.acc_lst_val.append(np.mean(self.acc_lst_val_tmp))

            ################# mean of each append to lst for plot###########
            ###########flush#############
            self.loss_lst_val_tmp = []
            self.acc_lst_val_tmp = []


        torch.set_grad_enabled(True)
        self.Datacon1Model.train()

    def TestStep(self):

        label_lst = ['5_b7_1', '1_00_0', '3_00_0', '3_b7_1', '6_a12_2', '4_00_0', '2_00_0', '5_a7_2', '6_00_0',
                          '5_b6_1', '3_b8_1', '2_a5_2', '6_a11_1', '3_b3_1', '3_a9_2', '3_a9_3', '3_a9_1', '5_00_0',
                          '6_b5_1', '5_b8_1', '3_b6_1', '6_b4_1', '6_a12_1', '6_b4_3', '6_a11_2']

        self.Datacon1Model.eval()
        countNum = 0
        self.optimizer.zero_grad()

        ResultLst = []

        with torch.set_grad_enabled(False):
            for ImageName,TestBInput in self.TestDataloader:


                TestBInput = (TestBInput.float()).to(self.device)

                TestBLogit = self.forward(TestBInput)
                TestBLogit = TestBLogit.cpu()

                arg_TestBLogit = torch.argmax(TestBLogit, dim=1).item()
                Total_pred = label_lst[arg_TestBLogit]

                ResultLst.append([str(ImageName[0]),Total_pred])
                print(f'{countNum} / {self.testLen} Pred done  data : {[str(ImageName[0]),Total_pred]}')
                countNum +=1


        ResultLst = sorted(ResultLst, key= lambda x: x[0])

        print(ResultLst)
        print('Start saving Result.....')


        header = ['image','label']
        with open(self.modelPlotSaveDir+'sample_submission.csv','w') as f:
            wr = csv.writer(f)
            wr.writerow(header)
            for i in ResultLst:
                wr.writerow(i)
                print(f'appending {i} complete')


        torch.set_grad_enabled(True)
        self.Datacon1Model.train()


    def START_TRN_VAL(self):

        for i in range(10000):
            print('training step start....')
            self.trainingStep(trainingNum=i)
            print('training step complete!')

            print('Validation start.....')
            self.valdatingStep(validatingNum=i)
            print('Validation complete!')


            fig = plt.figure()
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
            ax1.set_title('train loss')
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.plot(range(len(self.acc_lst_trn)), self.acc_lst_trn)
            ax2.set_title('train acc')

            ax3 = fig.add_subplot(1, 4, 3)
            ax3.plot(range(len(self.loss_lst_val)), self.loss_lst_val)
            ax3.set_title('val loss')
            ax4 = fig.add_subplot(1, 4, 4)
            ax4.plot(range(len(self.acc_lst_val)), self.acc_lst_val)
            ax4.set_title('val acc ')

            plt.savefig(self.modelPlotSaveDir +  'Result.png', dpi=300)
            print('saving plot complete!')
            plt.close()

            print(f'num4epoch is : {self.num4epoch} and self.max_epoch : {self.MaxEpoch}')

            self.num4epoch += 1
            if self.num4epoch >= self.MaxEpoch:
                break




if __name__ == '__main__':

    modelKind = 'resnet50'
    backboneOutFeature = 1000
    LinNum = 100
    totalCropNum = 25
    data_folder_dir_trn = '/home/a286winteriscoming/Downloads/Data4dacon1/data/train/'
    data_folder_dir_val  = '/home/a286winteriscoming/Downloads/Data4dacon1/data/val/'
    data_folder_dir_test = '/home/a286winteriscoming/Downloads/Data4dacon1/data/test/'
    MaxEpoch= 10
    iter_to_accumul = 2
    MaxStep = 20
    MaxStepVal = 10000
    bSizeTrn = 24
    save_range= 10
    modelLoadNum = 300

    savingDir = mk_name(model=modelKind,BckOutFt=backboneOutFeature,cNum=LinNum,bS=bSizeTrn)
    modelPlotSaveDir = '/home/a286winteriscoming/Downloads/Data4dacon1/'+savingDir + '/'
    createDirectory(modelPlotSaveDir)


    try:
        print(f'Loading {modelPlotSaveDir + str(modelLoadNum)}.pth')
        MODEL_START = torch.load(modelPlotSaveDir + str(modelLoadNum) + '.pth')
    except:
        MODEL_START  = Datacon1classifier(modelKind=modelKind,
                                          backboneOutFeature=backboneOutFeature,
                                          LinNum=LinNum,
                                          totalCropNum=totalCropNum,
                                          data_folder_dir_trn=data_folder_dir_trn,
                                          data_folder_dir_val=data_folder_dir_val,
                                          MaxEpoch=MaxEpoch,
                                          modelPlotSaveDir=modelPlotSaveDir,
                                          iter_to_accumul=iter_to_accumul,
                                          MaxStep=MaxStep,
                                          MaxStepVal=MaxStepVal,
                                          bSizeTrn= bSizeTrn,
                                          data_folder_dir_test= data_folder_dir_test,
                                          bSizeVal=10,lr=3e-4,eps=1e-9)

    #MODEL_START.TestStep()

    for i in range(10000):
        MODEL_START.START_TRN_VAL()

        if i%save_range ==0:
            if i > 15000:
                break

            try:
                torch.save(MODEL_START, modelPlotSaveDir + str(i) + '.pth')
                print('saving model complete')
                print('saving model complete')
                print('saving model complete')
                print('saving model complete')
                print('saving model complete')
                time.sleep(5)
            except:
                print('saving model failed')
                print('saving model failed')
                print('saving model failed')
                print('saving model failed')
                print('saving model failed')
                time.sleep(5)

























