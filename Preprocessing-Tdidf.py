# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 03:54:43 2020

@author: MMOHTASHIM
"""


from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import pandas as pd
import argparse
from tqdm import tqdm
import numpy as np

parser=argparse.ArgumentParser(description="Generating TDIDF from Given Annotations in the same order as images")
parser.add_argument('-p','--PATH',help="The Path for your main image data")
parser.add_argument('-s','--PATH_Filker',help="The Path for your Filker Pickles Dictionary")
parser.add_argument('-m','--max_features_tdidf',type=int,help="Max Features you want want for TDIDF Matrix")
args=parser.parse_args()


def main(path,filker_path,max_features_tdidf):
    pickles_dics=os.listdir(filker_path)
    df=pd.read_csv("results.csv", sep='|')
    df.set_index(df.columns.tolist()[0],inplace=True)
    annotations_corpus=[]
    
    for filker_dic in tqdm(pickles_dics):
        pickle_out = open(os.path.join(filker_path,filker_dic),"rb")
        filker_dic = pickle.load(pickle_out)
        for key_image in filker_dic.keys():
            annotations=df.loc[key_image+".jpg",df.columns.tolist()[1]].tolist()
            annotations_corpus.append(annotations)
    annotations_own_images=os.listdir(path)    
    for annotated in tqdm(annotations_own_images):
        if annotated!="flickr30k_images":
            for times in os.listdir(os.path.join(path,annotated)):
                annotations_corpus.append(annotated)
    np.save("Annotations_corpus.npy",np.array(annotations_corpus)) 
#    print(annotations_corpus)
    
    testing_tdidf_corpus=[]
    for instance in annotations_corpus:
            for sentence in instance:
                if len(str(sentence))>1:
                    testing_tdidf_corpus.append(sentence)
                else:
                    testing_tdidf_corpus.append(instance)
    tdidf=TfidfVectorizer(max_features=max_features_tdidf)##hamper with this
    tdidf.fit_transform(np.array(testing_tdidf_corpus))

    pickle_out = open(os.path.join(os.getcwd(),"tdidf_annotation.pickle"),"wb")
    pickle.dump(tdidf, pickle_out)
    pickle_out.close()
    
    np.save("Annotations_corpus.npy",np.array(annotations_corpus))
    print("Ran TDIDF and Saved the Annotations_corpus and TDIDF Trained")

def final_tdidf():
    pickle_out = open(os.path.join(os.getcwd(),"tdidf_annotation.pickle"),"rb")
    tdidf=pickle.load(pickle_out)
    pickle_out.close()
    
    Annotation_corpus=np.load("Annotations_corpus.npy",allow_pickle=True)
    final_tdidf_output=[]
    for instance in tqdm(Annotation_corpus):
            if isinstance(instance,list):
                for sentence in instance:
                    if pd.isnull(sentence):
                        del instance[instance.index(sentence)]
                new_instance=tdidf.transform(instance).toarray()
                new_instance=np.mean(new_instance,0)
                final_tdidf_output.append(new_instance.reshape(-1,))
            else:
                new_instance=tdidf.transform([instance]).toarray().reshape(-1,)
                final_tdidf_output.append(new_instance)
    final_tdidf_output=np.array(final_tdidf_output) 
    np.save("final_tdidf_output.npy",final_tdidf_output)          
    print(final_tdidf_output.shape)
                
            
                
                
            
    
    
    
    
   
    
    

if __name__=="__main__":
    main_dictionary=main(args.PATH,args.PATH_Filker,args.max_features_tdidf)
    final_tdidf()