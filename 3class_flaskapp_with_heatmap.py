"""
Flaskapp with heatmap
"""
import tensorflow as tf

import numpy as np

from api_aadhar_pan_passport_heatmap import api,GRADCAM

from flask import Flask  , request ,jsonify

from collections import defaultdict

app=Flask(__name__) 

modelArch=tf.keras.models.load_model('D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/model_mobilenet_v2_driving_license_all_4.h5')


gradModel=tf.keras.models.Model(
                inputs=[modelArch.input],
                outputs=[modelArch.get_layer('global_average_pooling2d_4').input,
                         modelArch.output]
                )
    
gradmodel=GRADCAM(modelArch,gradModel,layerName=None)

classnames=np.load('classnames_4.npy') 

@app.route('/tocode',methods=['POST'])
def getPredictions():
    finalPred=defaultdict(list)
    
    image=request.files.get("image").read()
    
    outputClassNames,outputCoords,outputconfidence,timeTaken=api(image,modelArch,gradmodel,classnames)
    
    if(outputconfidence[1]<85):
        finalPred[1]="Image unclear. Please input another image."
        finalPred[2]=outputconfidence[1]
    else:
        for i in outputClassNames.keys():
            finalPred[i].append([outputClassNames[i],
                             outputconfidence[i],
                             ])
        finalPred[4].append("Time taken: "+str(timeTaken))  
        finalPred[5].append(outputCoords)          
    return jsonify(finalPred)

if(__name__=='__main__'):
    app.run(port=5000)

