#region notes:
# dataset_path folder shall contain 3 folders named: training_data, test_data and validation_data
# each of those 3 folder shall contain 2 folders named: labels_0 and labels_1
# each image is resized to 128x128 size
# images num shall be divisible by batch size
# TODO: make it reads n(len(classes_names)) types of objects (each folder shall contain n folders (labels_i))
# TODO: takes certain new size of the images (maybe None for no resizing)
# TODO: images num has not to be divisible by batch size
# TODO: use property
#endregion

import  os
import glob
import random
from scipy.misc import imread, imsave
import cv2
import matplotlib.pyplot as plt
import numpy as np

class two_objects_dataset:

    def __init__(self,batch_size,dataset_path,classes_names):
        self.__classes_names=classes_names   # ['cats','dogs'] so label_1 folder contains dogs....
        self.__classes_num=len(classes_names)
        self.__batch_size=batch_size
        self.__dataset_path = dataset_path
        self.__training_data=[]
        self.__test_data=[]
        self.__validation_data=[]
        self.__training_ind=0
        self.__test_ind=0
        self.__validation_ind=0
        self.__read_training_data()
        self.__read_test_data()
        self.__read_validation_data()
        self.__shuffle_dataset()
        self.training_data_size=len(self.__training_data)
        self.test_data_size=len(self.__test_data)
        self.validation_data_size=len(self.__validation_data)

    def __read_training_data(self):
        print("Reading Training Data")
        # read labels 0
        path = os.path.join(self.__dataset_path + "/training_data/labels_0", '*g')  # g -> jpg
        files = glob.glob(path)
        for file in files:
            #img = imread(file)
            #cv2.resize(img,(128,128))
            #self.__training_data.append((img, 0))
            #read image , resize it , append it
            self.__training_data.append(  (cv2.resize(imread(file),(128,128)) , [1,0] )  )


        # read labels 1
        path = os.path.join(self.__dataset_path + "/training_data/labels_1", '*g')  # g -> jpg
        files = glob.glob(path)
        for file in files:
            #img = imread(file)
            self.__training_data.append(  (cv2.resize(imread(file),(128,128)) , [0,1] )  )

    def __read_test_data(self):
        print("Reading Test Data")
        # read labels 0
        path = os.path.join(self.__dataset_path + "/test_data/labels_0", '*g')  # g -> jpg
        files = glob.glob(path)
        for file in files:
            #img = imread(file)
            self.__test_data.append(  (cv2.resize(imread(file),(128,128)) , [1,0] )  )

        # read labels 1
        path = os.path.join(self.__dataset_path + "/test_data/labels_1", '*g')  # g -> jpg
        files = glob.glob(path)
        for file in files:
            #img = imread(file)
            self.__test_data.append(  (cv2.resize(imread(file),(128,128)) , [0,1] )  )

    def __read_validation_data(self):
        print("Reading Validation Data")
        # read labels 0
        path = os.path.join(self.__dataset_path + "/validation_data/labels_0", '*g')  # g -> jpg
        files = glob.glob(path)
        for file in files:
            #img = imread(file)
            self.__validation_data.append(  (cv2.resize(imread(file),(128,128)) , [1,0] )  )

        # read labels 1
        path = os.path.join(self.__dataset_path + "/validation_data/labels_1", '*g')  # g -> jpg
        files = glob.glob(path)
        for file in files:
            #img = imread(file)
            self.__validation_data.append(  (cv2.resize(imread(file),(128,128)) , [0,1] )  )

    def get_next_training_batch(self):
        #images=[]; labels=[]
        images=np.empty([self.__batch_size,128,128,3])
        labels=np.empty([self.__batch_size,2])
        ind_in_batch=0
        for i in range(self.__training_ind,self.__training_ind+self.__batch_size):
            img,lbl=self.__training_data[i]
            #images.append(img)
            #labels.append(lbl)
            images[ind_in_batch]=img
            labels[ind_in_batch]=lbl
            ind_in_batch+=1

        self.__training_ind += self.__batch_size
        if self.__training_ind == len(self.__training_data):   # self.training_data_size
            self.__training_ind = 0
        #images=np.asarray(images)
        return images,labels

    def get_next_test_batch(self):
        #images=[]; labels=[]
        images=np.empty([self.__batch_size,128,128,3])
        labels=np.empty([self.__batch_size,2])
        ind_in_batch=0
        for i in range(self.__test_ind, self.__test_ind+self.__batch_size):
            img,lbl=self.__test_data[i]
            #images.append(img)
            #labels.append(lbl)
            images[ind_in_batch]=img
            labels[ind_in_batch]=lbl
            ind_in_batch+=1

        self.__test_ind += self.__batch_size
        if self.__test_ind == self.test_data_size:
            self.__test_ind = 0
        return images,labels

    def get_next_validation_batch(self):
        images=[]; labels=[]
        for i in range(self.__validation_ind,self.__validation_ind+self.__batch_size):
            img,lbl=self.__validation_data[i]
            images.append(img)
            labels.append(lbl)

        self.__validation_ind += self.__batch_size
        if self.__validation_ind == self.validation_data_size:
            self.__validation_ind = 0
        return images,labels

    def __shuffle_dataset(self):
        random.shuffle(self.__training_data)
        random.shuffle(self.__test_data)
        random.shuffle(self.__validation_data)

    def fun(self):

        print(len(self.__training_data))
        print(len(self.__test_data))
        print(len(self.__validation_data))
        _,b=self.__training_data[0]
        print(b)
        _, b = self.__test_data[0]
        print(b)
        _, b = self.__validation_data[0]
        print(b)
        dog=0;cat=0
        for i in range(len(self.__test_data)):
            _,b=self.__test_data[i]
            if(b%2==0):
                dog+=1
            else:
                cat+=1
        print("Dogs num: {}   ,   Cats num: {}".format(dog,cat))
        print(_.shape)

        figure_obj = plt.figure()

        figure_obj.add_subplot(2, 2, 1).set_title('Original size')
        plt.imshow(_)

        _=cv2.resize(_,(128,128))
        figure_obj.add_subplot(2, 2, 2).set_title('128x128 size')
        plt.imshow(_)

        plt.show()



#region test read next patch, make that from main class
# for i in range(dataset.training_data_size//batch_size +3):          #3 or any num so exceeds one iteration
#     imgs,lbls=dataset.get_next_training_batch()                     #test data after one full iteration
#     if i == 0 or i == dataset.training_data_size // batch_size:     #are same? so it's faultless
#         print(i,": ",lbls)
#         print(i,": ",imgs[0].shape)
#
#         figure_obj = plt.figure()
#
#         figure_obj.add_subplot(3, 4, 1)
#         plt.imshow(imgs[0])
#         figure_obj.add_subplot(3, 4, 2)
#         plt.imshow(imgs[1])
#         figure_obj.add_subplot(3, 4, 3)
#         plt.imshow(imgs[2])
#         figure_obj.add_subplot(3, 4, 4)
#         plt.imshow(imgs[3])
#         figure_obj.add_subplot(3, 4, 5)
#         plt.imshow(imgs[4])
#         figure_obj.add_subplot(3, 4, 6)
#         plt.imshow(imgs[5])
#         figure_obj.add_subplot(3, 4, 7)
#         plt.imshow(imgs[6])
#         figure_obj.add_subplot(3, 4, 8)
#         plt.imshow(imgs[7])
#         figure_obj.add_subplot(3, 4, 9)
#         plt.imshow(imgs[8])
#         figure_obj.add_subplot(3, 4, 10)
#         plt.imshow(imgs[9])
#
#         plt.show()
#endregion
