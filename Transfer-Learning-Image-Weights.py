# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 01:29:08 2020

@author: MMOHTASHIM
"""

import pickle
import numpy 
from tensorflow import keras
from tensorflow.keras.applications.mobilenet import MobileNet
import argparse
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os
from tqdm import tqdm

parser=argparse.ArgumentParser(description="Encoding Image through InceptionResNetV2")
parser.add_argument('-p','--PATH',help="The Path for your Own Image Pickles Dictionary")
parser.add_argument('-s','--PATH_Filker',help="The Path for your Filker Pickles Dictionary")
args=parser.parse_args() 

def deep_encoder_model():
    base_model=MobileNet(include_top=True, weights='imagenet')
    model=Model(inputs=base_model.input,outputs=base_model.get_layer('conv_pw_13').output)##hamper with this
    print(model.summary())
    return model
    
def main(path_1,path_2):

    if not os.path.isdir(os.path.join(os.getcwd(),'dic_filker30k_images--encoded')):
        os.mkdir(os.path.join(os.getcwd(),'dic_filker30k_images--encoded'))
    if not os.path.isdir(os.path.join(os.getcwd(),'dic_my_images--encoded')):
        os.mkdir(os.path.join(os.getcwd(),'dic_my_images--encoded'))
    model=deep_encoder_model()
    filker_pickles=os.listdir(path_2)
    print(filker_pickles)
    own_images_pickles=os.listdir(path_1)
    for filker_pickle in tqdm(filker_pickles):
        encoded_filker_dic={}
        pickle_out = open(os.path.join(path_2,filker_pickle),"rb")
        filker_dic = pickle.load(pickle_out)
        for key_image in filker_dic.keys():
             encoded_version=model.predict(filker_dic[key_image].reshape(1,224,224,3))
             encoded_filker_dic[key_image]=encoded_version.reshape(7,7,1024)
        pickle_out = open(os.getcwd()+"\\dic_filker30k_images--encoded\\{}-encoded.pickle".format(filker_pickle[:-7]),"wb")
        pickle.dump(encoded_filker_dic, pickle_out)
        pickle_out.close()
    for own_image_pickle in tqdm(own_images_pickles):
        encoded_own_image_dic={}
        pickle_out = open(os.path.join(path_1,own_image_pickle),"rb")
        own_image_dic = pickle.load(pickle_out)
        for key_image in own_image_dic.keys():
             encoded_version=model.predict(own_image_dic[key_image].reshape(1,224,224,3))
             encoded_own_image_dic[key_image]=encoded_version.reshape(7,7,1024)
        pickle_out = open(os.getcwd()+"\\dic_my_images--encoded\\{}-encoded.pickle".format(own_image_pickle[:-7]),"wb")
        pickle.dump(encoded_own_image_dic, pickle_out)
        pickle_out.close()


if __name__=="__main__":
    main(args.PATH,args.PATH_Filker)
    
    
