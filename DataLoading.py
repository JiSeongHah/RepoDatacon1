import torch
import numpy as np
import os
from glob import glob
import json
from PIL import Image
from getDataFolders import getDataLst



class LoadData():
    def __init__(self,data_dir,TRAIN=True):
        self.data_dir = data_dir

        self.TRAIN = TRAIN

        if self.TRAIN == True:
            self.image_name,self.region,self.task,self.disease,\
            self.crop,self.area,self.grow,self.risk,\
            self.bbox = self.json_load(TRAIN=self.TRAIN)


    def img_load(self):
        img_arr = np.asarray(Image.open(self.data_dir+'.jpg'))

        if self.TRAIN == True:
            bbox_x,bbox_y,bbox_w,bbox_h = self.bbox['x'],self.bbox['y'],self.bbox['w'],self.bbox['h']
            croped_img = torch.from_numpy(img_arr[int(bbox_y):int(bbox_y+bbox_h),int(bbox_x):int(bbox_x+bbox_w),:])

        else:
            croped_img = torch.from_numpy(img_arr)

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

    def __init__(self,data_folder_dir,TRAIN=True):
        self.data_folder_dir = data_folder_dir

        self.data_folder_lst = os.listdir(data_folder_dir)

        self.TRAIN = TRAIN

        self.disrisk_label_lst = ['000', 'a111', 'a112', 'a121', 'a122', 'a52', 'a72', 'a91', 'a92', 'a93', 'b31', 'b41', 'b43',
                          'b51', 'b61', 'b71', 'b81']

    def __len__(self):
        return len(os.listdir(self.data_folder_dir))

    def __getitem__(self, idx):

        data_folder_name = self.data_folder_lst[idx]
        full_data_dir = self.data_folder_dir+data_folder_name+'/'+data_folder_name

        if self.TRAIN == True:
            input_tensor,label = LoadData(data_dir=full_data_dir,TRAIN=self.TRAIN).get_data_label()

            crop_label = float(label.split('_')[0])
            dis_risk_label = self.disrisk_label_lst.index(label.split('_')[1] + label.split('_')[2])

            return data_folder_name,input_tensor,crop_label,dis_risk_label

        else:
            input_tensor = LoadData(data_dir=full_data_dir, TRAIN=self.TRAIN).get_data_label()

            return data_folder_name,input_tensor






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