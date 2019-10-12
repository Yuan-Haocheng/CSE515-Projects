import os
import csv
import numpy as np
from skimage.io import imread_collection
import phase1_main

#--------------------------------------------------------
# files : a list of file's names
# label : metadata : 'right' || 'left' || 'dorsal' || 'palmar' || 'with' || 'without' || 'male' || 'female'
# csv_infor : csv file containing metadata

def get_matrix(files,csv_infor,label,folder_name):
    img_meta_list = []
    filenames = []
    selected_img = []
    index = value = 0
    for rows in csv_infor:                              # check given data set's metadata
        for fn in files:
            if fn==rows[7]:
                filenames.append(fn)
                # find
                if rows[6]=='dorsal right':             # parameters: a: 1-right hand,0-left hand
                    a=b=1                               # parameters: b: 1-dorsal, 0-palmar
                elif rows[6]=='palmar right':           # parameters: c: 1-with accessories,0-without accessories
                    a=1                                 # parameters: d: 1-male,0-female
                    b=0
                elif rows[6] == 'dorsal left':
                    a=0
                    b=1
                else :
                    a=b=0
                if rows[2] =='male':
                    d=1
                else :
                    d=0

                img_meta_list.append([a,b,int(rows[5]),d])   #list for image-metadata

    # construct index and value for searching images with the given label
    if label == 'right':
        index = 0
        value = 1
    elif label == 'left':
        index = 0
        value = 0
    elif label == 'dorsal':
        index = 1
        value = 1
    elif label == 'palmar':
        index = 1
        value = 0
    elif label == 'with':
        index = 2
        value = 1
    elif label == 'without':
        index = 2
        value = 0
    elif label == 'male':
        index = 3
        value = 1
    elif label == 'female':
        index = 3
        value = 0
    for i in range(len(filenames)):
        if img_meta_list[i][index] == value:
            selected_img.append(folder_name+'/'+filenames[i])
    return selected_img



#--------------------------------------------------------
# files : a list of file's names
# model : model selected : 'LBP' || 'CM' || 'HOG' || 'SIFT'
# label : metadata : 'right' || 'left' || 'dorsal' || 'palmar' || 'with' || 'without' || 'male' || 'female'
# tech : dimension reduced tech : 'PCA' || 'SVD' || 'NMF' || 'PCA'
# k : value of k ( k-top latent semantics)
# csv_infor : csv file containing metadata


def task_3(folder_name,csvname,model,label,tech,k):

    for hahaha in os.walk(folder_name):      # given dataset
        hahaha[2].sort()                                  # by number order
    files = hahaha[2]
    csv_infor = csv.reader(open(csvname, 'r'))  # load hand information, csv format
    img_list = get_matrix(files,csv_infor,label,folder_name)
    img_meta = np.array(img_list)
    colln_imgs = (imread_collection(img_meta))
    print(colln_imgs)
# choose different model
# X will be nxm array ( n images, m features)
    if model == 'CM':
        X = phase1_main.moments(colln_imgs)
    elif model == 'LBP':
        X = phase1_main.LBP(colln_imgs)
    elif model == 'HOG':
        X = phase1_main.HOG(colln_imgs)
    elif model == 'SIFT':
        X = phase1_main.SIFT(colln_imgs)

#choose different tech

    if tech == 'PCA':
        U,V = phase1_main.PCA(X,k)
    elif tech == 'SVD':
        U,V = phase1_main.SVD(X,k)
    elif tech == 'NMF':
        U,V = phase1_main.aNMF(X, k)
    elif tech == 'PCA':
        U,V = phase1_main.LDA(X, k)
    print(U, V)





    # need visualization now


if __name__ == "__main__":


                                        # ignore an empty list
    model='LBP'
    tech = 'NMF'
    k = 2
    #files is a list of sorted filenames
    label = 'right'
    path = 'small'
    csvname = 'handinfo.csv'
    task_3(path,csvname,model,label,tech,k)

