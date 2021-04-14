#정규화를 위해 계산한번해야함

import os
import cv2
import numpy as np

# root_path =r'face_detecting_data\crawler\imageset'
# imageset_list = os.listdir(root_path)

# BGR_mean = np.zeros((3), np.float32)
# BGR_std = np.zeros((3), np.float32)
# image_count = 0
# for emotion in imageset_list:
    
#     emotion_image_path = os.path.join(root_path, emotion)
#     emotion_images = os.listdir(emotion_image_path)
#     for image in emotion_images:
#         image = cv2.imread(os.path.join(emotion_image_path, image))

#         image_count += 1
#         BGR_mean[0] += np.mean(image[:,:,0])
#         BGR_mean[1] += np.mean(image[:,:,1])
#         BGR_mean[2] += np.mean(image[:,:,2])
#         BGR_std[0] += np.std(image[:,:,0])
#         BGR_std[1] += np.std(image[:,:,1])
#         BGR_std[2] += np.std(image[:,:,2])

# print(f'mean : {BGR_mean / image_count}, std : {BGR_std / image_count}')

import torch
import torch.nn
import random
from torch.utils.data import Dataset, DataLoader

class Custom_Dataloder(torch.utils.data.Dataset):
    def __init__(self):
        one_hot = {
            'Anger'   : [1,0,0,0,0],
            'Disgust' : [0,1,0,0,0],
            'Fear'    : [0,0,1,0,0],
            'Joy'     : [0,0,0,1,0],
            'Sadness' : [0,0,0,0,1],
        }

        root_path =r'face_detecting_data\crawler\imageset'
        imageset_list = os.listdir(root_path)
        train_dataset = []
        for emotion in imageset_list:
            emotion_image_path = os.path.join(root_path, emotion)
            emotion_images = os.listdir(emotion_image_path)
            for image in emotion_images:
                image = cv2.imread(os.path.join(emotion_image_path, image))
                # resize
                image = cv2.resize(image, (64,64))
                # BGR2Gray
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # normalize
                image = (image[:,:] - 127.5) / 127.5

                train_dataset.append([image, one_hot[emotion]])
        random.shuffle(train_dataset)
        train_dataset = np.array(train_dataset)
        
        self.x_data = train_dataset[:,0]
        self.y_data = train_dataset[:,1]
        

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.shape[0]


dataset = Custom_Dataloder()
trian_loder = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
for i, data in enumerate(trian_loder):
    input, label = data
    print(f'{input.shape}')

