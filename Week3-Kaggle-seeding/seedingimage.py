#
# author: sting.huang(arihuang@hotmail.com)
# 2017.12.14
#


import re, os, random
import numpy as np
from PIL import Image
import time

CLASS_NUM = 12
IMG_H = 128
IMG_W = 128

class dataset:
    def __init__(self, path, save_resize = False):
        self.dataPool = [] #[[label, image]]
        self.labelMap = dict()
        st = time.time()
        self.load_image(path, save_resize)
        print(self.labelMap)
        print("Total data set:", len(self.dataPool), np.shape(self.dataPool), " time:", time.time() - st)

    def load_image( self, path, save_resize) :
        subpaths = os.listdir(path)
        if len(subpaths) != CLASS_NUM:
            print("error! subpath: {0} != class number: {1}".format(len(subpaths), CLASS_NUM))
            return
        for labelCNT, labelName in zip(range(CLASS_NUM), subpaths):
            self.labelMap[labelCNT] = labelName
            label_oh = np.zeros((CLASS_NUM,))
            label_oh[labelCNT] = 1
            for fname in os.listdir(path + '/'+ labelName):
                imgdata = Image.open(path + '/'+ labelName +'/'+ fname).convert('RGB')
                if np.shape(imgdata)[2] != 3:
                    print(fname, " => ", np.shape(imgdata))
                h,w = imgdata.size
                if h != IMG_H or w != IMG_W:
                    print(" need to resize :", h, w)
                    imgdata = imgdata.resize(( IMG_H,IMG_W),  Image.BILINEAR)
                    if save_resize:
                        imgdata.save(path + '/'+ labelName +'/'+ fname)
                #print(fname, np.shape(imgdata), type(imgdata), type(imgdata.getdata()))
                self.dataPool.append( [label_oh,np.array(imgdata.getdata(),dtype= np.float32).reshape(IMG_H, IMG_W, 3), imgdata])
            print("{0}/{1} dataset loaded".format(labelCNT+1, CLASS_NUM))

    def get_batch(self, size):
        images = []
        labels = []
        for data in random.sample( self.dataPool, size):
            images.append(data[1])
            labels.append(data[0])
        #print(np.shape(images), np.shape(labels))
        return images, labels

    def get_images(self, size):
        images_np = []
        images = []
        labels = []
        for data in random.sample( self.dataPool, size):
            images_np.append(data[1])
            images.append(data[2])
            labels.append(data[0])
        #print(np.shape(images), np.shape(labels))
        return images_np, labels, images

    def mapLabel(self, code):
        return self.labelMap[np.argmax(code)]

if __name__ == "__main__":
    db = dataset("train", False)
    db.get_batch(16)
