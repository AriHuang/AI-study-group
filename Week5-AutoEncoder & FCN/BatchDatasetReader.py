import numpy as np
import scipy.misc as misc
import os, random, re


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, data_path, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param data_path: dataset path
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        filters = []
        Categories = dict()
        if image_options.get('filter', None) != None:
            filters = [re.compile(x) for x in image_options['filter']]
            f = open('dataset/sceneCategories.txt')
            for line in f:
                k, v = line.replace('\n', '').split(' ')
                Categories[k] = v
        if len(filters) == 0:
            files = [f for f in os.listdir(data_path+ "/images")]
        else:
            files = []
            for f in os.listdir(data_path+ "/images"):
                c = Categories[f.replace('.jpg', '')]
                for ft in filters:
                    if ft.search(c) != None:
                        files.append(f)
        image_files = [data_path+ "/images/" + f for f in files]
        annotation_files = [data_path+ "/annotations/"+ f.replace('.jpg', '.png') for f in files]
        self.image_options = image_options
        self.__channels = True
        self.images = np.array([self._transform(f) for f in image_files])
        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform(f), axis=3) for f in annotation_files], dtype= np.int32)
        print (self.images.shape)
        print (self.annotations.shape)



    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])
        if self.image_options.get("resize", False):
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image
        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_random_batch(self, batch_size):
        indexes = random.sample( range(self.images.shape[0]), batch_size)
        #indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

if __name__ == '__main__':
    IMAGE_SIZE = 224
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE, 'filter': ['room', 'kitchen']}
    BatchDatset("dataset/training", image_options = image_options)