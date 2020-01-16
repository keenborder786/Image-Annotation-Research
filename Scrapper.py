# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 19:08:46 2020

@author: MMOHTASHIM
"""

from icrawler.builtin import GoogleImageCrawler
import os
from nltk.corpus import words

class image_scrapper(object):
    def __init__(self,image_annontation,max_images,directory):
        self.image_annotation=image_annontation
        self.max_images=max_images
        self.directory=directory
    def scrap(self):
        image_path=os.path.join(self.directory,self.image_annotation)
        print(image_path)
        if not os.path.isdir(image_path):
            os.makedirs(image_path)
        google_crawler = GoogleImageCrawler(storage={'root_dir': image_path})
        google_crawler.crawl(keyword=self.image_annotation, max_num=self.max_images)



if __name__=="__main__":
    text1=["Cat on Table","Pizza on Table","Dog on Table",
          "Kids Playing Together","Men and Women Kissing",]
    text2=["Couples Dancing Together","Laptop Laying Over",
           "BeautiFull Mountains","Young Boy Sitting Next to Tree"]
    text3=["boys playing together","girls playing together","Childer going to school bus",
           "Children Studying in school","Farmer in farm","Family eating food"]
    text1=["Cat on Table","Pizza on Table","Dog on Table",
          "Kids Playing Together","Men and Women Kissing",]
    text2=["Couples Dancing Together","Laptop Laying Over",
           "BeautiFull Mountains","Young Boy Sitting Next to Tree"]
    text3=["boys playing together","girls playing together","Childer going to school bus",
           "Children Studying in school","Farmer in farm","Family eating food"]
    food=["pineapple is on the apple","boy is eating pizza","girl is eating pizza",
          "Family is eating pork","Cat is eating its food","dog is eating its food",
          "boy is drinking pepsi","girl is drinking pepsi"]
    nature=["A very beautiful mountain","A very beautifull park","A very vast sea",
            "A very awesome valley","Children are playing in park","Tourist are enjoying near park",
            "Tourist are enjoying near mountains","A very beautifull waterfall",
            "Sunset","Sunrise","Full MoonLight","Autumn Season","Spring Season","There is SnowFall","Sun is Shining"]
    sports=["Boys playing football","Girls Playing Football","Boys Playing Cricket","Girls Playing Cricket",
            "Fans are enjoying baseball","Fans are enjoying cricket","Tennis Game is on","Olympics are being held","Swimming Competition has started"]
    adult=["Sexy women","Sexy Men","Sexy Women in Binkini","Sexy Men in Underwear","Sexy Beautiful and Young Women"]
    home=["huge home","small home","hut"]
    for query in home:
        scraper=image_scrapper(query,20,os.path.join(os.getcwd(),'data'))
        scraper.scrap()