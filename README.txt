

model1.h5 - Trained with images without augmentation
model2.h5 - Trained with images with augmentation
model_mobilenet_v2.h5 - Trained using MobileNetV2 with augmentation
model_mobilenet_v2_passport.h5- Trained for 3 classes with augmentation
model_mobilenet_v2_half_passport.h5- Trained with half passports (3 classes)
model_mobilenet_v2_driving_license_all_4.h5 - All 4 classes with augmentation

.npy files:

1. classnames_2.npy - 2 classes (aadhar and pan)
2. classnames_3.npy - 3 classes(aadhar,pan and passport)
3.classnames_4.npy - 4 classes 

files:

1. api_aadhar_pan_passport_heatmap.py - Heatmap + 3 classes
2. api_aadhar_vs_pancard.py - 3 classes (without heatmap)
3. Augmentation_for_2_classes.py - For doing augmentation
4. 3class_flaskapp_with_heatmap.py - 3 classes flaskapp + heatmap
5. aadhar_pan_passport_classification.py - Training file for the 3 classes
6. aadhar_pan_passport_flaskapp.py - 3 class flaskapp without heatmap
7. Aadhar_vs_PAN_classification.py - Training file for 2 classes
8. aadhar_vs_pancard_flaskapp.py - 2 class flaskapp without heatmap

Folders:
Outputs: Images with rect bounding box drawn via heatmap(3 class with full passport)
Output_1: For visual check for 3 class (half passport)
Output_2 : For visual check for 4 class
Old_passport_images - Contains full passport and half passport images as well