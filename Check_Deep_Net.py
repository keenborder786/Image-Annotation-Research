# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 00:07:13 2020

@author: MMOHTASHIM
"""


from tensorflow.keras.models import load_model,Model
import argparse
from tensorflow.keras.applications.mobilenet import MobileNet
import os 
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

parser=argparse.ArgumentParser(description="Predict the annontation for a give image")
parser.add_argument('-p','--PATH',help="The Path for images that you want to test")
parser.add_argument('-d','--model_name',help="Name of your model")
args=parser.parse_args()

def deep_encoder_model():
    base_model=MobileNet(include_top=True, weights='imagenet')
    model=Model(inputs=base_model.input,outputs=base_model.get_layer('conv_pw_13').output)
#    print(model.summary())
    return model

def main_predicting_pipeline_deep_net(PATH,model_name):
    testing_images=os.listdir(PATH)
    model=load_model(model_name)
    pickle_out = open(os.path.join(os.getcwd(),"tdidf_annotation.pickle"),"rb")
    tdidf=pickle.load(pickle_out)
    pickle_out.close()
    corpus=np.load("Annotations_corpus.npy",allow_pickle=True)
    tdidf_matrix=np.load("final_tdidf_output.npy")
    print(testing_images)
    for image_one in testing_images:
        image_path=PATH+'\{}'.format(image_one)
        image_array=image.img_to_array(image.load_img(image_path,target_size=(224,224))).reshape(1,224,224,3)
        
        Mobile_Net_encoder=deep_encoder_model()
        encoded_image=Mobile_Net_encoder.predict(image_array)
        
        predicted_tdidf_encoding=model.predict(encoded_image)

        distances=[]
        for tdidf_vector in tdidf_matrix:
            distances.append(np.linalg.norm(tdidf_vector-predicted_tdidf_encoding))
#        print(np.array(distances).shape)
 
        max_dist=0
        for distance in distances:
            if distance>max_dist:
                max_dist=distance
                max_distance=distances.index(distance)
        
        print(corpus[max_distance])
        break

if __name__=="__main__":
    main_predicting_pipeline_deep_net(args.PATH,args.model_name)
        
        
        
    
    