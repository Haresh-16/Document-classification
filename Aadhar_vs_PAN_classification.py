"""
Aadhar card vs PAN card classification
""" 

import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50 , MobileNetV2
import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from PIL import Image #To processing binary image from form-data POSTMAN
import io # For decoding image input from POSTMAN

from glob import glob
from collections import defaultdict




def load_images(path:str):
    
    """
    Loads images and labels into memory
    """
    
    train_ds =tf.keras.preprocessing.image_dataset_from_directory(
            directory=path,
            validation_split=0.1,
            labels="inferred",
            batch_size=32,
            subset="training",
            image_size=(128,128),
            shuffle=True,
            interpolation="bilinear",
            seed=42
            )
    
    valid_ds=tf.keras.preprocessing.image_dataset_from_directory(
            directory=path,
            validation_split=0.1,
            labels="inferred",
            batch_size=32,
            subset="validation",
            image_size=(128,128),
            shuffle=True,
            interpolation="bilinear",
            seed=42
            )
    
    print("Image iterators loaded!")
    
    return train_ds , valid_ds

def dataset_viz(train_ds,class_names):
    
    plt.figure(figsize=(10,10))
    for images,labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3,3,i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classnames[labels[i]])
            plt.axis("off")
    
def feature_extraction(imgs,input_shape):
    """
    Return features extracted from ResNet50
    
    More efficient to store features and then apply new model on it
    """
    extracted_features=[]
    basemodel=ResNet50(include_top=False,input_shape=input_shape,pooling='avg')
    for i in imgs:
        input_img=np.expand_dims(i,axis=0)
        extracted_features.append(basemodel(input_img))
    return np.array(extracted_features)

def model(input_shape):
    """
    Returns model architecture
    """
    
    basemodel=ResNet50(include_top=False,input_shape=input_shape)
    
    #Freeze the model
    basemodel.trainable=False
    
    #PIPELINE STARTS HERE!!
    inputs=tf.keras.Input(shape=input_shape)
    
    #Data Augmentation
    
    x=tf.keras.applications.resnet.preprocess_input(inputs)
    
    """
    #Normalization
    norm_layer=tf.keras.layers.experimental.preprocessing.Normalization()
    mean=np.array([127.5]*3)
    var=mean**2
    #Scale inputs to [-1,+1]
    x=norm_layer(x)
    norm_layer.set_weights([mean,var])
    """
        
    x=basemodel(x,training=False)
    
    x=keras.layers.GlobalAveragePooling2D()(x)
    
    x=keras.layers.Dense(100,
                        activation='relu',
                        kernel_initializer='glorot_normal',
                        use_bias=True)(x)
    
    outputs=keras.layers.Dense(2,
                               use_bias=True,
                               kernel_initializer='glorot_normal')(x)
    
    model=keras.Model(inputs,outputs)
    
    print("Model constructed!")
        
    return model

def model_mobilenetV2(input_shape):
    """
    Returns model architecture
    """
    
    basemodel=MobileNetV2(include_top=False,input_shape=input_shape)
    
    #Freeze the model
    basemodel.trainable=False
    
    #PIPELINE STARTS HERE!!
    inputs=tf.keras.Input(shape=input_shape)
    
    #Data Augmentation
    
    x=tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    """
    #Normalization
    norm_layer=tf.keras.layers.experimental.preprocessing.Normalization()
    mean=np.array([127.5]*3)
    var=mean**2
    #Scale inputs to [-1,+1]
    x=norm_layer(x)
    norm_layer.set_weights([mean,var])
    """
        
    x=basemodel(x,training=False)
    
    x=keras.layers.GlobalAveragePooling2D()(x)
    
    # x=keras.layers.Dense(100,
    #                     activation='relu',
    #                     kernel_initializer='glorot_normal',
    #                     use_bias=True)(x)
    
    outputs=keras.layers.Dense(2,
                               use_bias=True,
                               kernel_initializer='glorot_normal')(x)
    
    model=keras.Model(inputs,outputs)
    
    print("Model constructed!")
        
    return model

def preprocesslabels(labels):
    
    onehotencoder=OneHotEncoder()
    labels=np.reshape(labels,(-1,1))
    yLabels=onehotencoder.fit_transform(labels)
    print(onehotencoder.get_params)
    print("Preprocessing of labels done!")
    return np.array(yLabels.toarray())
    

def train(model,imgs,labels):
    """
    Train
    """
    #labels=preprocesslabels(labels)
            
    #trainX,testX,trainY,testY=train_test_split(imgs,labels,test_size=0.25,
     #                                          stratify=labels)
    
    
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4,
                                       name='Adam')
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print("Model Training started!")

    model.fit(train_ds,
              validation_data=valid_ds,              
              epochs=20,
              callbacks=keras.callbacks.EarlyStopping(
                      monitor='val_loss',
                      patience=6,
                      mode='min',
                      ))
    
def evaluate():
    pass
    
def saveModel():
    pass

def optimizePipeline(train_ds,valid_ds):
    
    AUTOTUNE=tf.data.experimental.AUTOTUNE
    
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    
    valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    return train_ds,valid_ds

@tf.function(experimental_relax_shapes=True)
def inference(modelArch,image,classnames):
    
    pred=modelArch(image)
    
    predIndex=tf.argsort(pred[0])[-2:][::-1]
    
    confi=tf.sort(tf.nn.softmax(pred[0]))[-2:][::-1]
    
    predIndexUnpacked=tf.unstack(tf.reshape(predIndex,[-1]))
    
    #tf.print([classnames[i] for i in map(int,predIndexUnpacked)]," - ",confi*100)
    
    return predIndex,pred,predIndexUnpacked,confi

"""VALIDATING IMAGES"""

def validating_imgs(data_dir):
    """
    Parameters
    ----------
    data_dir : str
        Directory where images are stored.

    Returns
    -------
    None.

    """
    for i in os.listdir('D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Dataset'):
        for j in glob(os.path.join('D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Dataset',i,"*.*")):
            img=Image.open(j)
            val=img.load()
            print("\n")
            print(j,":",val)


#VALIDATING IMAGES DONE!
#validating_imgs('D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Dataset')

if(__name__=='__main__'):
    
    
    path=r'D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Aug_dataset'
    
    input_shape=(128,128,3)
    
    train_ds ,valid_ds = load_images(path)
    
    classnames=train_ds.class_names
    
    np.save(r'D:\Tocode.ai task 1\New_CNN_for_aadhar_pancard_classification\classnames.npy',classnames) #HERE!!
    
    #dataset_viz(train_ds,classnames)
    
    #==OPTIMIZED DATASET ITERATORS OBTAINED====#
    train_ds,valid_ds=optimizePipeline(train_ds,valid_ds)
    
    #modelArch=model(input_shape) #ResNet50 model
    
    model_mobilenetV2=model_mobilenetV2(input_shape)
    
    train(model_mobilenetV2,train_ds,valid_ds)
    
    print("TRAINING OVER!!")
    
    model_mobilenetV2.save("model_mobilenet_v2.h5") #Model saved here!
    
    imagepath='D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Test_images/pan_out_of_dist.jpg'
    
    tfImg=tf.convert_to_tensor(np.expand_dims(tf.keras.preprocessing.image.load_img(
            path=imagepath,
            target_size=(128,128),
           interpolation='bilinear'),axis=0))
    
    predIndex,pred,predIndexUnpacked,confi=inference(model_mobilenetV2,tfImg,classnames)
    
    for i in predIndexUnpacked:
        print(classnames[np.array(i)])
        
    print("\n")
        
    for i in confi:
        print(np.array(i)*100,"%")
    
    """
    DO test set classification report
    """


    
        














    
    
    
    
