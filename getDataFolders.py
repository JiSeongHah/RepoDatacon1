from glob import glob
import os
import torch
import json
import numpy as np
from PIL import Image

class LoadData():
    def __init__(self,data_dir,TRAIN=True):
        self.data_dir = data_dir

        self.TRAIN = TRAIN

        if self.TRAIN == True:
            self.image_name,self.region,self.task,self.disease,\
            self.crop,self.area,self.grow,self.risk,\
            self.bbox = self.json_load(TRAIN=self.TRAIN)

        self.label_lst = ['000', 'a111', 'a112', 'a121', 'a122', 'a52', 'a72', 'a91', 'a92', 'a93', 'b31', 'b41', 'b43', 'b51', 'b61', 'b71', 'b81']

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


def getDataLst(path):
    return os.listdir(path)


# path = '/home/a286winteriscoming/Downloads/Data4dacon1/data/train/'
# lst = getDataLst(path)
# print(lst)
#
#
#
#
# label_lst = []
#
# for i in lst:
#     _,label = LoadData(path+i+'/'+i).get_data_label()
#
#     #label_kind =label.split('_')[1]+label.split('_')[2]
#
#     label_kind = label.split('_')[0]
#
#     print(label_kind)
#
#     if label_kind not in label_lst:
#         label_lst.append(label_kind)
#
# print(len(label_lst))
# print(sorted(label_lst))