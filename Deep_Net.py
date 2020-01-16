# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:05:40 2020

@author: MMOHTASHIM
"""
import os
import pickle
import numpy as np 
import argparse
from tensorflow.keras.layers import Dense,Input,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.models import Model
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from tqdm import tqdm

parser=argparse.ArgumentParser(description="Opening the Encoded Image pickle and feed into algorithm")
parser.add_argument('-p','--PATH',help="The Path for your filker encoded image data")
parser.add_argument('-i','--PATH_2',help="The Path for your own encoded image data")
parser.add_argument('-c','--Chunks',type=int,help="How many chunks from filker_30k")##hamper with this
args=parser.parse_args()

def open_data(path,path_2,Chunks):
    total_dic_filker=os.listdir(path)
    total_dic_own_image=os.listdir(path_2)
    total_image=[]
    for i in tqdm(range(Chunks)):
         filker_dic=total_dic_filker[i]
         pickle_out = open(os.path.join(path,filker_dic),"rb")
         filker_dic=pickle.load(pickle_out)
         pickle_out.close()
         for images_key in filker_dic.keys():
            total_image.append(filker_dic[images_key])
            
            
    for own_image_dic in tqdm(total_dic_own_image):
        pickle_out = open(os.path.join(path_2,own_image_dic),"rb")
        own_image_dic=pickle.load(pickle_out)
        pickle_out.close()
        for image_key in own_image_dic.keys():
            total_image.append(own_image_dic[image_key])
    
    
    
    total_image=np.array(total_image)
    print(total_image.shape)
    np.save('X.npy',total_image)
            
        
def DEEP_NETWORK():
    X=np.load("X.npy")
    y=np.load("final_tdidf_output.npy")
    y=y[:X.shape[0]]
    log_dir=os.getcwd()+ "\\logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    inputs=Input(shape=(X.shape[1:]),name="input")
    
    conv_2d_layer_1=Conv2D(32,(3,3),name="conv_2d_layer_1",activation="relu")(inputs)
    conv_2d_layer_2=Conv2D(32,(2,2),name="conv_2d_layer_2",activation="relu")(conv_2d_layer_1)
    conv_2d_layer_3=Conv2D(32,(1,1),name="conv_2d_layer_3",activation="relu")(conv_2d_layer_2)
    
    
    Max_pool_2d_layer_1=MaxPool2D((2,2),strides=(1,1))(conv_2d_layer_3)
    Dense_layer=Flatten()(Max_pool_2d_layer_1)
    
    fc_1=Dense(1024,activation="relu")(Dense_layer)##hamper with this
    fc_2=Dense(512,activation="relu")(fc_1)
    fc_3=Dense(1024,activation="relu")(fc_2)
    
    y_pred=Dense(y.shape[1],activation="linear")(fc_3)
    model=Model(inputs=inputs,outputs=y_pred)
    checkpoint=ModelCheckpoint(os.getcwd()+"\\Deep_Net--Version-1-TDIDF.h5",monitor='loss', save_best_only=False)
    earlystop=EarlyStopping(monitor='loss', min_delta=0, patience=3)
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    
    
    
    model.compile(optimizer="Adam",loss="mean_squared_error")
    model.fit(X,y,callbacks=[tensorboard_callback,earlystop,checkpoint],epochs=50)
    


    

if __name__=="__main__":
    open_data(args.PATH,args.PATH_2,args.Chunks)
    print("Now Running the Neural Network")
    DEEP_NETWORK()
    
    
    
    