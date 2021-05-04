from PIL import Image 
import PIL
import os 
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image as im
import random

def resize_image(filename):
    img = cv2.imread(filename)
    img = cv2.resize(img,(150,150))
    return img

def get_SIFT_features(images):
    sift = cv2.SIFT_create()
    descriptors_ind = np.empty((0,128), int)
    descriptors = []
    for image in images:
        step_size = 10
        kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, image.shape[0], step_size) 
                                    for x in range(0, image.shape[1], step_size)]
        _,desc = sift.compute(image, kp)
        descriptors_ind = np.vstack((descriptors_ind, desc))
        descriptors.append(desc)
    descriptors = np.array(descriptors)
    return descriptors_ind, descriptors


def get_images(folder_name,resize=False):
    images = []
    labels = []
    files = os.listdir(folder_name)
    # for folder in my_list:
        # if not folder.startswith("."):
            # files = os.listdir(folder_name+"/"+folder)
    for file in files:
        if not file.startswith("."):
            if resize:
                img = resize_image(folder_name+"/"+file)
            else:
                img = cv2.imread(folder_name+"/"+file)
            images.append(img)
            l = file.split("_")
            labels.append(l[0])
    return images, labels
    
def get_kmeans(n_clusters,descriptors_ind):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(descriptors_ind)
    return kmeans

def get_histogram(kmeans, descriptors, n_clusters, images):
        
    hist_values = np.zeros((len(images),n_clusters), int)
    
    for i in range(len(images)):
        for j in range(len(descriptors[i])):
            cluster = kmeans.predict([descriptors[i][j]])
            hist_values[i][cluster[0]]+=1
            
    return hist_values


def get_target_image(test_images,path):
    n_clusters = 300
    
    images, labels = get_images(path, resize=True)

    descriptors_ind, descriptors = get_SIFT_features(images)

    ## get kmeans model
    kmeans = get_kmeans(n_clusters,descriptors_ind)

    ## calculate histogram of visual words
    hist_values = get_histogram(kmeans,descriptors,n_clusters,images)
        
    ## calculating tf
    tf = hist_values/hist_values.sum(axis=1, keepdims=True)

    ##calculating idf
    N = len(images)
    dfi = (hist_values != 0).sum(0)
    dfi= np.log(N/dfi)
        
    ## get tf_idf
    tf_idfs = np.multiply(tf,dfi)

    ## training SVM
    clf = OneVsRestClassifier(SVC(kernel="linear"))
    clf.fit(tf_idfs, labels)

    temp = [test_images]
    test_descriptors_ind, test_descriptors = get_SIFT_features(temp)
    ## get histogram of visual words for test data
    test_hist_values = get_histogram(kmeans,test_descriptors,n_clusters,temp)
        
    ## calculating tf
    tf_test = test_hist_values/test_hist_values.sum(axis=1, keepdims=True)
    
    ## calculate tf-idf for test data
    tf_idfs_test = np.multiply(tf_test,dfi)
    
    ## predicting the labels 
    predicted_labels = clf.predict(tf_idfs_test)

    return predicted_labels[0]