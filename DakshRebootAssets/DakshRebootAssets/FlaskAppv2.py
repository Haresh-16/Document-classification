"""
Flask Server for FlaskAppV2
"""
from flask import Flask , render_template , Response , request ,redirect ,jsonify
import cv2
import tensorflow as tf
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from glob import glob
from collections import defaultdict

from DakshRebootNetwork_4_with_SpaCy import GRADCAM , api , inference #Contains the core APIs 

app=Flask(__name__) 

"""Loading models and other dependencies"""
modelArch=tf.keras.models.load_model('DakshRebootModel_2.h5')

gradModel=tf.keras.models.Model(
                inputs=[modelArch.input],
                outputs=[modelArch.get_layer('global_average_pooling2d').input,
                         modelArch.output]
                )
    
gradmodel=GRADCAM(modelArch,gradModel,layerName=None)

classnames=np.load('classnames.npy') 

@app.route('/dakshHit',methods=['POST'])
def getPredictions():
    finalPred=defaultdict(list)
    inputtext=request.form.get('text')
    image=request.files.get("image").read() #Absolute path of image , which is input to the model
    outputClassNames,Top2coords,outputconfidence,finalTextInference=api(image,modelArch,gradmodel,classnames,inputtext)
    # ^^ Dict of top 2 classnames returned from api(). Two overlaid images will be saved to ..\Outputs dir.
    
    for i in outputClassNames.keys():
        finalPred[i].append([outputClassNames[i],Top2coords[i],str(outputconfidence[i]*100)+"%"+" Confidence",finalTextInference])
    return jsonify(finalPred) #jsonify(outputClassNames) , jsonify(Top2coords) ,jsonify(outputconfidence) # outputClassNames will have classnames in order of preference. Top2Coords too , are in order of preference!
    
if(__name__=='__main__'):
    app.run(port =5000)
