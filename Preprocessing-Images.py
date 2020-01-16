# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 16:30:51 2020

@author: MMOHTASHIM
"""

import argparse
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tqdm import tqdm
import pickle

parser=argparse.ArgumentParser(description="Preprocessing the Images into an dictionary and storing in chunks")
parser.add_argument('-p','--PATH',help="The Path for your main image data")
parser.add_argument('-i','--Image',type=int,help="How many images to store in each chunk")
args=parser.parse_args()


def main(path,image_number):
    files=os.listdir(path)
    if not os.path.isdir(os.path.join(os.getcwd(),'dic_filker30k_images')):
        os.mkdir(os.path.join(os.getcwd(),'dic_filker30k_images'))
    if not os.path.isdir(os.path.join(os.getcwd(),'dic_my_images')):
        os.mkdir(os.path.join(os.getcwd(),'dic_my_images'))
    for folder in tqdm(files):
        if folder=="flickr30k_images":##this will execute only once
            images_all=os.listdir(path+'\{}'.format(folder))
            len_images_all=len(images_all)
            index_jump=image_number
            prev_index=0
            for index in range(index_jump,len_images_all,index_jump):
                images_all_index=images_all[prev_index:index]
                filker_dic={}
                for image_one in images_all_index:
                    image_path=path+'\{}'.format(folder)+'\{}'.format(image_one)
                    image_array=image.img_to_array(image.load_img(image_path,target_size=(224,224)))
                    filker_dic[image_one[:-4]]=image_array
                prev_index+=index_jump
                pickle_out = open(os.getcwd()+"\\dic_filker30k_images\\folder--{}-{}.pickle".format(folder,prev_index),"wb")
                pickle.dump(filker_dic, pickle_out)
                pickle_out.close()
            images_all_index_final=images_all[index:len_images_all]
            filker_dic={}
            for image_one in images_all_index_final:
                    image_path=path+'\{}'.format(folder)+'\{}'.format(image_one)
                    image_array=image.img_to_array(image.load_img(image_path,target_size=(224,224)))
                    filker_dic[image_one[:-4]]=image_array
            pickle_out = open(os.getcwd()+"\\dic_filker30k_images\\folder--{}-{}.pickle".format(folder,len_images_all),"wb")
            pickle.dump(filker_dic, pickle_out)
            pickle_out.close()
        else:
            own_images={}
            images_all=os.listdir(path+'\{}'.format(folder))
            for image_one in images_all:
                    image_path=path+'\{}'.format(folder)+'\{}'.format(image_one)
                    image_array=image.img_to_array(image.load_img(image_path,target_size=(224,224)))
                    own_images[folder]=image_array  
            pickle_out = open(os.getcwd()+"\\dic_my_images\\folder--{}.pickle".format(folder),"wb")
            pickle.dump(own_images, pickle_out)
            pickle_out.close()


if __name__=="__main__":
    main_dictionary=main(args.PATH,args.Image)
    
