#
# author: sting.huang(arihuang@hotmail.com)
# 2017.12.14
#


import random
import numpy as np

CLASS_NUM = 10
IMG_H = 32
IMG_W = 32

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data_one(f):
    print("load image from ..", f)
    batch = unpickle(f)
    data = np.reshape(batch[b'data'] , [-1, 3, IMG_H, IMG_W])
    data = data.transpose([0, 2, 3, 1])
    labels = batch[b'labels']
    return data, labels


class cifar_dataset:
    def __init__(self, path = "cifar-10"):
        meta = unpickle(path + '/batches.meta')
        self.labels_name = meta[b'label_names']
        train_files = [ 'data_batch_{}'.format(d) for d in range(1, 6) ]
        self.valid = []
        self.valid_label = []
        self.train = []
        self.train_label = []
        for f in train_files[1:]:
            data, labels = load_data_one(path+'/'+ f)
            self.train.extend(list(data))
            labels_onehot = np.zeros((len(labels), CLASS_NUM))
            for n in range(len(labels)):
                labels_onehot[n , labels[n]] = 1
            self.train_label.extend(labels_onehot)
        data, labels = load_data_one(path+'/'+ train_files[0])
        self.valid = list(data)
        self.valid_label = np.zeros((len(labels), CLASS_NUM))
        for n in range(len(labels)):
            self.valid_label[n , labels[n]] = 1


    def get_batch(self, size):
        images = []
        labels = []
        for img, lab in random.sample( list(zip(self.train, self.train_label)) , size):
            images.append(img)
            labels.append(lab)
        return images, labels

    def get_valid(self, size = 512):
        images = []
        labels = []
        for img, lab in random.sample( list(zip(self.valid, self.valid_label)) , size):
            images.append(img)
            labels.append(lab)
        return images, labels

    def map_name(self, id):
        return self.labels_name[id]


if __name__ == '__main__':
    dataset = cifar_dataset()
    img, lab = dataset.get_batch(64)
    print(np.shape(img), np.shape(lab))
