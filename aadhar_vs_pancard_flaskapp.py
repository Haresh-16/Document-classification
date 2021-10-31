import tensorflow as tf

import numpy as np

from api_aadhar_vs_pancard import api

from flask import Flask  , request ,jsonify

from collections import defaultdict

app=Flask(__name__) 

modelArch=tf.keras.models.load_model('D:/Tocode.ai task 1/New_CNN_for_aadhar_pancard_classification/model_mobilenet_v2.h5')

classnames=np.load('classnames.npy') 

@app.route('/tocode',methods=['POST'])
def getPredictions():
    finalPred=defaultdict(list)
    
    image=request.files.get("image").read()
    
    outputClassNames,outputconfidence,timeTaken=api(image,modelArch,classnames)
    
    for i in outputClassNames.keys():
        finalPred[i].append([outputClassNames[i],
                             outputconfidence[i],
                             ])
    finalPred[3].append("Time taken: "+str(timeTaken))            
    return jsonify(finalPred)

if(__name__=='__main__'):
    app.run(port=5000)

