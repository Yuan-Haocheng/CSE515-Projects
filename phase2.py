import os
import numpy as np
import csv
from sklearn.decomposition import NMF



def task_8(csv_infor,files):




    img_meta_list = []

    for rows in csv_infor:                              # check given data set's metadata
        for fn in files:
            if fn==rows[7]:                             # find
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

                img_meta_list.append((a,b,int(rows[5]),d))   #list for image-metadata


    img_meta = np.array(img_meta_list)      # image-metadata matrix -> nx4
    print(img_meta)


    k = 2

    nmf = NMF(n_components=k,  # k value
              # init=None,  # initial method for W H，including 'random' | 'nndsvd'(default) |  'nndsvda' | 'nndsvdar' | 'custom'.
              # solver='cd',  # 'cd' | 'mu'
              # beta_loss='frobenius',  # {'frobenius', 'kullback-leibler', 'itakura-saito'}
              # tol=1e-4,  #condition for stopping
              # max_iter=200,  # max times for itreration
              # random_state=None,
              # alpha=0.,  # Regularization parameter
              # l1_ratio=0.,  # Regularization parameter
              # verbose=0,  # Lengthy mode
              # shuffle=False  # for "cd solver"
              )

    X = img_meta
    X = X.T  # transpose X
    nmf.fit(X)  # run NMF
    W = nmf.fit_transform(X)  # get matrix W
    H = nmf.components_  # matrix H



    # print output
    print('metadata space:')
    for ki in range(k):
        print('k' + str(ki) + ' = ', end=' ')
        output = []
        dic = {}
        for i in range(4):
            dic[W[i, ki]] = i
            output.append((W[i, ki], i))  # index
        output.sort(key=lambda x: x[0], reverse=1)              #decrease
        print(output)                                           # kx4 matrix, each element is weight-index pair

    print('image space:')
    H = H.T
    for ki in range(k):

        print('k' + str(ki) + ' = ', end=' ')
        output = []
        dic = {}
        for i in range(len(files)):
            dic[H[i, ki]] = i
            output.append((H[i, ki], files[i]))  # index
        output.sort(key=lambda x: x[0], reverse=1)              #decrease
        print(output)                                           # kxn       each element is weight-image pair





csv_infor = csv.reader(open('handinfo.csv','r'))    # load hand information, csv format
for file in os.walk('/Users/tiancaikening/PycharmProjects/sperate/small'):      # given dataset
    file[2].sort()                                  # by number order
files = file[2]                                     # ignore an empty list

#files is a list of sorted filenames


task_8(csv_infor,files)