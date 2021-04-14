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