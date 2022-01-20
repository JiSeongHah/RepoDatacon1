import torch
import numpy as np
import os
from glob import glob
import json
from PIL import Image
from getDataFolders import getDataLst
from torchvision import transforms


class LoadData():
    def __init__(self,data_dir,size4res,TRAIN=True,CROP=True):
        self.data_dir = data_dir

        self.TRAIN = TRAIN

        self.CROP = CROP

        self.size4res = size4res

        self.transforms = transforms.Compose([
                                              transforms.Resize(size=(self.size4res[0],
                                                                      self.size4res[1])),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              transforms.RandomVerticalFlip(p=0.5),

                                              ])

        self.transformsNocrop = transforms.Compose([
            transforms.Resize(size=(self.size4res[0], self.size4res[1])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),

        ])

        self.transformsTest = transforms.Compose([
            transforms.Resize(size=(self.size4res[0], self.size4res[1])),
        ])

        self.transformsTestNocrop = transforms.Compose([
            transforms.Resize(size=(self.size4res[0], self.size4res[1])),
        ])



        if self.TRAIN == True:
            self.image_name,self.region,self.task,self.disease,\
            self.crop,self.area,self.grow,self.risk,\
            self.bbox = self.json_load(TRAIN=self.TRAIN)


    def img_load(self):
        img_arr = np.asarray(Image.open(self.data_dir+'.jpg'))

        if self.TRAIN == True:
            if self.CROP == True:
                bbox_x,bbox_y,bbox_w,bbox_h = self.bbox['x'],self.bbox['y'],self.bbox['w'],self.bbox['h']
                croped_img = torch.from_numpy(img_arr[int(bbox_y):int(bbox_y+bbox_h),int(bbox_x):int(bbox_x+bbox_w),:])
                croped_img = croped_img.permute(2,0,1)
                croped_img = self.transforms(croped_img).float()

            if self.CROP != True:
                croped_img = torch.from_numpy(img_arr)
                croped_img = croped_img.permute(2, 0, 1)
                croped_img = self.transformsNocrop(croped_img).float()

        else:
            if self.CROP == True:
                croped_img = torch.from_numpy(img_arr)
                croped_img = croped_img.permute(2, 0, 1)
                croped_img = self.transformsTest(croped_img)
            if self.CROP != True:
                croped_img = torch.from_numpy(img_arr)
                croped_img = croped_img.permute(2, 0, 1)
                croped_img = self.transformsTestNocrop(croped_img)


        return croped_img

    def json_load(self,TRAIN=True):

        if TRAIN == True:
            self.data_json_path = self.data_dir+'.json'

            with open(self.data_json_path,'r') as json_file:
                data_json = json.load(json_file)

                image_name = data_json['description']['image']

                region = data_json['description']['region']

                task = data_json['description']['task']

                disease = data_json['annotations']['disease']

                crop = data_json['annotations']['crop']

                area = data_json['annotations']['area']

                grow = data_json['annotations']['grow']

                risk = data_json['annotations']['risk']

                bbox = data_json['annotations']['bbox'][0]

            return image_name,region,task,disease,crop,area,grow,risk,bbox

        if TRAIN == False:

            return

    def get_data_label(self):

        if self.TRAIN == True:
            input = self.img_load()

            label = str(self.crop)+'_'+str(self.disease)+'_'+str(self.risk)

            return input,label

        else:
            input = self.img_load()

            return input





class MyDatacon1Dataset(torch.utils.data.Dataset):

    def __init__(self,data_folder_dir,size4res,TRAIN=True,CROP=True):
        self.data_folder_dir = data_folder_dir

        self.data_folder_lst = os.listdir(data_folder_dir)

        self.TRAIN = TRAIN

        self.CROP = CROP

        self.size4res = size4res

        self.label_lst = ['5_b7_1', '1_00_0', '3_00_0', '3_b7_1', '6_a12_2', '4_00_0', '2_00_0', '5_a7_2', '6_00_0',
                          '5_b6_1', '3_b8_1', '2_a5_2', '6_a11_1', '3_b3_1', '3_a9_2', '3_a9_3', '3_a9_1', '5_00_0',
                          '6_b5_1', '5_b8_1', '3_b6_1', '6_b4_1', '6_a12_1', '6_b4_3', '6_a11_2']

    def __len__(self):
        return len(os.listdir(self.data_folder_dir))

    def __getitem__(self, idx):

        data_folder_name = self.data_folder_lst[idx]
        full_data_dir = self.data_folder_dir+data_folder_name+'/'+data_folder_name

        if self.TRAIN == True:
            input_tensor,label = LoadData(data_dir=full_data_dir,TRAIN=self.TRAIN,CROP=self.CROP,size4res=self.size4res).get_data_label()

            label = self.label_lst.index(label)

            return data_folder_name,input_tensor,label

        else:
            input_tensor = LoadData(data_dir=full_data_dir, TRAIN=self.TRAIN,CROP=self.CROP,size4res=self.size4res).get_data_label()

            return data_folder_name,input_tensor



# path = '/home/a286winteriscoming/Downloads/Data4dacon1/data/train/'
#
# dt= MyDatacon1Dataset(data_folder_dir=path,TRAIN=True)
# labelLst = []
#
#
# for idx,i in enumerate(dt):
#     print(f'{idx} th done')
#     print(i[2])
#     label = i[2]
#     if label not in labelLst:
#         labelLst.append(label)
#
# print(labelLst)
# print(len(labelLst))
#
# wLst= []
# hLst = []
#
# cnum= 0
# for i in dt:
#     wLst.append(i[1].shape[1])
#     hLst.append(i[1].shape[0])
#     print(wLst[-1],hLst[-1])
#     print(f'{cnum} done')
#     cnum += 1
#
# print(f'mean of w is : {np.mean(wLst)} and mean of h is : {np.mean(hLst)} ')


#
#
# if __name__ == '__main__':
#
#     t_dta_dir = '/home/a286winteriscoming/Downloads/Data4dacon1/data/train/'
#
#     test_json = t_dta_dir + '10348' + '/10348.json'
#
#     with open(test_json, 'r') as json_file:
#         json_data = json.load(json_file)
#
#     # print(json_data['annotations']['bbox'][0])
#
#     x = json_data['annotations']['bbox'][0]['x']
#     y = json_data['annotations']['bbox'][0]['y']
#     w = json_data['annotations']['bbox'][0]['w']
#     h = json_data['annotations']['bbox'][0]['h']
#
#     print(json_data)
#     xx = json_data['annotations']['part'][0]['x']
#     yy = json_data['annotations']['part'][0]['y']
#     ww = json_data['annotations']['part'][0]['w']
#     hh = json_data['annotations']['part'][0]['h']
#
#     print(x, y, w, h)
#     img = t_dta_dir + '10348' + '/10348.jpg'
#
#     image_arr = np.asarray(Image.open(img))
#     print(image_arr.shape)
#
#     croped_img = image_arr[int(y):int(y + h), int(x):int(x + w), :]
#     part_croped_img = image_arr[int(yy):int(yy + hh), int(xx):int(xx + ww), :]
#
#     print(croped_img.shape)
#     print(part_croped_img.shape)
#
#     re_img = Image.fromarray(croped_img)
#     re_img.show()
#
#     re_part_img = Image.fromarray(part_croped_img)
#     re_part_img.show()