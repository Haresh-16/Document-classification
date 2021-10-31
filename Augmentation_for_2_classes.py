"""
Augmenting images for 2 class classification
"""

import albumentations as A

import cv2

from glob import glob

import os

transform = A.Compose([
        A.Transpose(),
        A.OneOf([
            A.GaussNoise()
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        A.OneOf([
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=.1)
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.RandomBrightnessContrast()            
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
    ])

#For Aadhar card
for i in glob("D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/New_Dataset/New_aadhar_dataset/*.*"):
    count=15
    image=cv2.imread(i)
    basename=os.path.basename(i)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    while(count>=0):
        transformed_image=transform(image=image)["image"]
        cv2.imwrite(os.path.join("D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Aug_dataset/Aadhar_card",basename.split('.')[0]+str(count)+'.jpg'),
                transformed_image)
        count-=1
        
#For PAN card
for i in glob("D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/New_Dataset/New_PAN_dataset/*.*"):
    count=15
    image=cv2.imread(i)
    basename=os.path.basename(i)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    while(count>=0):
        transformed_image=transform(image=image)["image"]
        cv2.imwrite(os.path.join("D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Aug_dataset/PAN_card",basename.split('.')[0]+str(count)+'.jpg'),
                transformed_image)
        count-=1
        
#For passport
for i in glob("D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/passport_images/*.*"):
    count=4
    image=cv2.imread(i)
    basename=os.path.basename(i)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    while(count>=0):
        transformed_image=transform(image=image)["image"]
        cv2.imwrite(os.path.join("D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Aug_dataset_passport_included/Passport",basename.split('.')[0]+str(count)+'.jpg'),
                transformed_image)
        count-=1

#For Driving license
for i in glob("D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Driving_license_data/*.*"):
    count=20
    image=cv2.imread(i)
    basename=os.path.basename(i)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    while(count>=0):
        transformed_image=transform(image=image)["image"]
        cv2.imwrite(os.path.join("D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Aug_dataset_passport_included/Driving_license",basename.split('.')[0]+str(count)+'.jpg'),
                transformed_image)
        count-=1

