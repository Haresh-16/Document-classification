"""
Final API for Daksh Reboot + SpaCy keyword from text ( api() is Configured to match with Flask Appv2.py )

Note:
    Outputs folder is created (line 21) to hold the overloid images!
""" 

import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from PIL import Image #To processing binary image from form-data POSTMAN
import io # For decoding image input from POSTMAN

#For spaCy keyword extraction!
import spacy
from collections import Counter
from string import punctuation
import en_core_web_md
from collections import defaultdict

nlp = en_core_web_md.load()

if(os.path.exists('Outputs') is False):
    os.mkdir('Outputs') 


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
            image_size=(100,100),
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
            image_size=(100,100),
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
    
    #data augmentation pipeline
    data_augmentation=keras.Sequential(
            [
                    tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",seed=42),
                    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1,seed=42)
            ])
    
    basemodel=ResNet50(include_top=False,input_shape=input_shape)
    
    #Freeze the model
    basemodel.trainable=False
    
    #PIPELINE STARTS HERE!!
    inputs=tf.keras.Input(shape=input_shape)
    
    #Data Augmentation
    x=data_augmentation(inputs)
    
    x=tf.keras.applications.resnet.preprocess_input(x)
    
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
    
    outputs=keras.layers.Dense(6,
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
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))

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

#========FOR GRADCAM VIZ============#

class GRADCAM:
    
    @tf.function
    def __init__(self,model,gradmodel,layerName):
        self.model=model
        self.gradmodel=gradmodel
        self.layerName=layerName
        
    @tf.function(experimental_relax_shapes=True)
    def compute_heatmap(self,inputs,classIdx,eps=1e-8):
        
        with tf.GradientTape() as tape:
            [conv_outputs,predictions]=self.gradmodel(inputs)
            loss=predictions[:,classIdx]
        
        grads=tape.gradient(loss,conv_outputs)
            
	# compute the guided gradients
        castConvOutputs = tf.cast(conv_outputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")

        guidedGrads = castConvOutputs * castGrads * grads

	# the convolution and guided gradients have a batch dimension
	# (which we don't need) so let's grab the volume itself and
	# discard the batch
        conv_outputs = conv_outputs[0]
        guidedGrads = guidedGrads[0]
        
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
                
		# grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
        
        
        (w, h) = (inputs.shape[2], inputs.shape[1])
        heatmap = preprocessing.Resizing(w, h)(tf.expand_dims(cam,axis=0))
        heatmap=heatmap[0,:,:]
        
	# normalize the heatmap such that all values lie in the range
	# [0, 1], scale the resulting values to the range [0, 255],
	# and then convert to an unsigned 8-bit integer

        numer = heatmap - tf.keras.backend.min(heatmap)
        denom = (tf.keras.backend.max(heatmap) - tf.keras.backend.min(heatmap)) + eps
        heatmap = numer / denom
        heatmap = tf.cast(heatmap * 255,"uint8")
		# return the resulting heatmap to the calling function
        
        return heatmap,loss
    
    def overlay_heatmap(self , heatmap, image,alpha=0.5,
        colormap=cv2.COLORMAP_VIRIDIS):
        
        heatmap=cv2.applyColorMap(heatmap,colormap)
        output=cv2.addWeighted(image,alpha,heatmap,1-alpha,0)
        
        return (heatmap,output)

#if(__name__=='__main__'):
    
    #modelArch=tf.keras.models.load_model('E:\DakshReboot\DakshRebootModel_2.h5')

    #gradModel=tf.keras.models.Model(
#                inputs=[modelArch.input],
              #  outputs=[modelArch.get_layer('global_average_pooling2d').input,
             #            modelArch.output]
              #  )
    
    #gradmodel=GRADCAM(modelArch,gradModel,layerName=None)
    
    #path=r'E:\DakshReboot\Classes'
    
    #input_shape=(100,100,3)
    
    #train_ds ,valid_ds = load_images(path)
    
    #classnames=train_ds.class_names
    
    #np.save('E:\DakshReboot\classnames.npy',classnames) #HERE!!
    
  
    #dataset_viz(train_ds,classnames)
    
    #==OPTIMIZED DATASET ITERATORS OBTAINED====#
    #train_ds,valid_ds=optimizePipeline(train_ds,valid_ds)
    
    #modelArch=model(input_shape)
    
    #train(modelArch,train_ds,valid_ds)
    
    #print("TRAINING OVER!!")
    
    #imagepath='E:\DakshReboot\Test images\over_5.jpg'
    
    #tfImg=tf.convert_to_tensor(np.expand_dims(tf.keras.preprocessing.image.load_img(
            #path=imagepath,
           # target_size=(100,100),
           # interpolation='bilinear'),axis=0))
    
    #Toppredindex,allpred=inference(modelArch,tfImg,classnames)
    
global count # Suffix for output images!

def finalTextComplaint(result , classnamesPOS):
    
    for i in result:
        for j in classnamesPOS.keys():
            for k in classnamesPOS[j]:
                if(i==k):
                    return j
        
def api(image,modelArch,gradmodel,classnames,text:str):
    """

    Parameters
    ----------
    image : binary image from post request via POSTMAN of type werkzeug.datastructures.FileStorage.
    
        Image input through Postman.
    
    text : Complaint description from User

    Returns
    -------
    Dict of top-2 predictions' classnames and their bbox coords.

    """
    
    classnames=np.load('classnames.npy') #HERE!!
    
    classnamesPOS=defaultdict(list)
    
    finalTextInference=""
    
    for i in classnames:
        classnamesPOS[i].extend(i.split(" "))
    
    result=[] # Processed input text of complaint description!
    
    POSTAGS=['PROPN','ADJ','NOUN']
    
    doc=nlp(text.lower())
    
    for token in doc:
        
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            
            continue
        
        if(token.pos_ in POSTAGS):
            
            result.append(token.text)
    
    # Finalising input complaint from Complaint maker!
    finalTextInference=finalTextComplaint(result , classnamesPOS)
                    
    outputClassNames={}
    
    outputCoords={}
    
    outputconfidence={}
    
    global count 
    
    count=1
    
    readImg=Image.open(io.BytesIO(image)).convert('RGB')
    readImg=np.array(readImg)
    readImg=readImg[:,:,::-1].copy()
    

    tfImg=tf.convert_to_tensor(np.expand_dims(cv2.resize(readImg,(100,100)),axis=0))

    tfImgCompHeatMapInput=tf.cast(tfImg,tf.float32)

    predIdxes,Allpred,predIndexUnpacked,confi=inference(modelArch,tfImg,classnames)

    for i in map(int,predIndexUnpacked):
        print(classnames[i])

    predIdxes=tf.convert_to_tensor(predIdxes)

    print("For ",predIdxes)

    outputList=[]

    for v,i in enumerate(map(int,predIdxes)):

        heatmap,loss=gradmodel.compute_heatmap(inputs= tfImgCompHeatMapInput,
                                       classIdx=i
                                       )
        
        heatmap = cv2.resize(heatmap.numpy(),(readImg.shape[1], readImg.shape[0]))

        _,overlayed=gradmodel.overlay_heatmap(heatmap,
                                      image= readImg
                                      )

#=====DRAW BBOX ON HIGH ACTIVATION AREA======#

        coords_req = np.where(heatmap>=0.75*np.max(heatmap))

        coords_req=list(zip(coords_req[0],coords_req[1]))

        dum=np.zeros((readImg.shape[0],readImg.shape[1]),dtype="uint8")

        for j in coords_req:
            dum[j[0],j[1]]=255 #Thresholded image is now obtained!
    
        cnts,_= cv2.findContours(dum.copy(),
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)


        for c in cnts:
            x,y,w,h=cv2.boundingRect(c)
            output_img=cv2.rectangle(readImg.copy(),
                            (x,y),
                            (x+w,y+h),
                            (0,0,255),
                            2
                  )
            #if(list(map(float,confi))[v]*100 > 75):
            cv2.imwrite('\Outputs\outImg'+str(count)+'.jpg',output_img)
            outputClassNames[count]=classnames[i]
            outputCoords[count]=[x,y,w,h]
            outputconfidence[count]=list(map(float,confi))[v]
            count+=1
                
                
    return outputClassNames,outputCoords,outputconfidence,finalTextInference
    #Returning the dict of output classnames to be rendered at final.html
    



#=====IMPLEMENT CCL FROM SCRATCH AND TEST ON A OVERLAID IMAGE===#


#===TO BE DONE TODAY (27/10/2020)=========#

    
        














    
    
    
    
