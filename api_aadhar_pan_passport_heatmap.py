"""
API for flask app ( Aadhar card vs Pan card classification with heatmap )
"""
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
import time

from tensorflow.keras.layers.experimental import preprocessing


import os

if(os.path.exists('Outputs_2') is False):
    os.mkdir('Outputs_2')

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

@tf.function(experimental_relax_shapes=True)
def inference(modelArch,image,classnames):
    
    pred=modelArch(image)
    
    predIndex=tf.argsort(pred[0])[-4:][::-1]
    
    confi=tf.sort(tf.nn.softmax(pred[0]))[-4:][::-1]
    
    predIndexUnpacked=tf.unstack(tf.reshape(predIndex,[-1]))
    
    #tf.print([classnames[i] for i in map(int,predIndexUnpacked)]," - ",confi*100)
    
    return predIndex,pred,predIndexUnpacked,confi

def api(image,modelArch,gradmodel,classnames):
    """
    Parameters
    ----------
    image : binary image from post request via POSTMAN of type werkzeug.datastructures.FileStorage.
        Image input through Postman.
    """
    outputClassNames={}
    
    outputconfidence={}
    
    outputCoords={}
    
    timeTaken=None
    
    count=1
    
    readImg=Image.open(io.BytesIO(image)).convert('RGB')
    readImg=np.array(readImg)
    readImg=readImg[:,:,::-1].copy()
    

    tfImg=tf.convert_to_tensor(np.expand_dims(cv2.resize(readImg,(128,128)),axis=0))
    
    tfImgCompHeatMapInput=tf.cast(tfImg,tf.float32)
    
    start=time.time()
    
    predIdxes,Allpred,predIndexUnpacked,confi=inference(modelArch,tfImg,classnames)
    
    timeTaken=time.time()-start
    
    print("Time taken:",timeTaken)
    
    for i in predIndexUnpacked:
        if(classnames[np.array(i)]=='PAN_card'):
            outputClassNames[count]='PAN card'#classnames[np.array(i)]
        elif(classnames[np.array(i)]=='Aadhar_card'):
            outputClassNames[count]='Aadhar card'
        elif(classnames[np.array(i)]=='Passport'):
            outputClassNames[count]='Passport'
        elif(classnames[np.array(i)]=='Driving_license'):
            outputClassNames[count]='Driving license'
        count+=1
    
    count=1
    
    for i in confi:
        outputconfidence[count]=np.array(i)*100
        count+=1
    
    
    print(outputClassNames)
    print(outputconfidence)
    
    count=1
    
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
                            thickness=5
                  )
            #if(list(map(float,confi))[v]*100 > 75):
            cv2.imwrite('D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/Outputs_2/outImg5'+str(count)+'.jpg',output_img)
            #cv2.imshow("output"+str(count),output_img)
            #outputClassNames[count]=classnames[i]
            outputCoords[count]=[x,y,w,h]
            #outputconfidence[count]=list(map(float,confi))[v]
            count+=1
    
    return outputClassNames,outputCoords,outputconfidence,timeTaken
        
        
    
    

