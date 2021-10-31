"""
API for flask app ( Aadhar card vs Pan card classification )
"""
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import cv2
import time

@tf.function(experimental_relax_shapes=True)
def inference(modelArch,image,classnames):
    
    pred=modelArch(image)
    
    predIndex=tf.argsort(pred[0])[-4:][::-1]
    
    confi=tf.sort(tf.nn.softmax(pred[0]))[-4:][::-1]
    
    predIndexUnpacked=tf.unstack(tf.reshape(predIndex,[-1]))
    
    #tf.print([classnames[i] for i in map(int,predIndexUnpacked)]," - ",confi*100)
    
    return predIndex,pred,predIndexUnpacked,confi

    
def api(image,modelArch,classnames):
    """
    Parameters
    ----------
    image : binary image from post request via POSTMAN of type werkzeug.datastructures.FileStorage.
        Image input through Postman.
    """
    outputClassNames={}
    
    outputconfidence={}
    
    timeTaken=None
    
    count=1
    
    readImg=Image.open(io.BytesIO(image)).convert('RGB')
    readImg=np.array(readImg)
    readImg=readImg[:,:,::-1].copy()
    

    tfImg=tf.convert_to_tensor(np.expand_dims(cv2.resize(readImg,(128,128)),axis=0))
    
    start=time.time()
    
    predIdxes,Allpred,predIndexUnpacked,confi=inference(modelArch,tfImg,classnames)
    
    timeTaken=time.time()-start
    
    print("Time taken:",timeTaken)
    
    # for i in predIndexUnpacked:
    #     if(classnames[np.array(i)]=='PAN_card'):
    #         outputClassNames[count]='PAN card'#classnames[np.array(i)]
    #     elif(classnames[np.array(i)]=='Aadhar_card'):
    #         outputClassNames[count]='Aadhar card'
    #     elif(classnames[np.array(i)]=='Passport'):
    #         outputClassNames[count]='Passport'
    #     elif(classnames[np.array(i)]=='Driving_license'):
    #         outputClassNames[count]='Driving license'
    #     count+=1
    
    print("predIndexUnpacked: ",predIndexUnpacked)
    
    for i in predIndexUnpacked:
        outputClassNames[count]=classnames[np.array(i)]
        count+=1
    
    count=1
    
    for i in confi:
        outputconfidence[count]=np.array(i)*100
        count+=1
    
    
    print(outputClassNames)
    print(outputconfidence)
    
    return outputClassNames,outputconfidence,timeTaken
        
        
    
    

