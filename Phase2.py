import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import pickle
import os
from scipy.stats import skew
from skimage.io import imread_collection, imread, imshow, show, imshow_collection
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2yuv, rgb2gray
from skimage.transform import resize
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from cv2 import xfeatures2d_SIFT
import joblib
import cv2
import json
import pandas
from collections import Counter


def getImgID(PATH):
    '''
    Function to get names of the JPG images in the input path in alphabetical order
    To get names of images of different format, change '.jpg' to the required format
    Input: PATH to the folder containing images
    Output: List of size N, where N is the number of images in the folder
    '''

    imgID = []
    for file in sorted(os.listdir(PATH)):
        if '.jpg' in file:
            file = file.split('.')[0]
            file = file.split('_')[1]
            imgID.append(file)
    return imgID


def loadImgs(PATH):
    '''
    Function to load and read the JPG images in the input path in alphabetical order of their names
    To read images of different format, change '.jpg' to the required format
    Input: PATH to the folder containing images
    Output: List of size N, where N is the number of images in the folder
    '''

    return(imread_collection(PATH+'*.jpg'))


def getMetadata(PATH, imgID):
    '''
    Function to get metadata of the images
    Input: PATH: path to the folder containing HandsInfo.csv file
           imgID: List of image names of length N where N is the number of images
    Output: metaData: Array of size (N, 9) where N is the number of images and each column corresponds to an attribute
    '''
    
    metadata = pandas.read_csv(PATH+'HandInfo.csv').values
    imgIDs = list(metadata[:,7])
    idx = [imgIDs.index('Hand_'+i+'.jpg') for i in imgID]
    metaData = metadata[idx,:]
    return metaData


def getImgCorrLabel(imgs, metaData, l):
    '''
    Function to get images corresponding to the label
    Input: imgs: List of images of length N
           metaData: Array of size (N, 9)
           l: Label 
    Output: imgCorrLabel: List of images corresponding to the label
            idxCorrLabel: List of indices of the images corresponding to the label
    '''

    imgCorrLabel = []
    idxCorrLabel = []
    imgs = np.array(imgs)
    if l=='left' or l=='right' or l=='dorsal' or l=='palmar':
        for i,v in enumerate(list(metaData[:,6])):
            if l in v:
                idxCorrLabel.append(i)
                imgCorrLabel.append(imgs[i])
    elif l=='male' or l=='female':
        for i,v in enumerate(list(metaData[:,2])):
            if l==v:
                idxCorrLabel.append(i)                
                imgCorrLabel.append(imgs[i])
    elif l=='wacc':
        for i,v in enumerate(list(metaData[:,4])):
            if v==1:
                idxCorrLabel.append(i)
                imgCorrLabel.append(imgs[i])
    else:
        for i,v in enumerate(list(metaData[:,4])):
            if v==0:
                idxCorrLabel.append(i)
                imgCorrLabel.append(imgs[i])  
    return imgCorrLabel, idxCorrLabel             


def euclideanDist(f1, f2):
    '''
    Function to calculate Euclidean Distance between two features
    Input: Two features of equal length
    Output: A real value >= 0
    '''

    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(f1, f2)]))


def cosineDist(f1, f2):
    '''
    Function to calculate Cosine Distance between two features
    Input: Two features of equal length
    Output: A real value in [0,100]%
    '''

    dot = np.dot(f1, f2)
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    return ((1 + (dot / (norm1 * norm2)))/2)*100


def moments(colln_imgs):
    '''
    Function to compute color moments of list of images
    Input: List of length N, where N is the number of images
    Output: Numpy Array of size (N, M), where N is the number of images and M = 1728
    '''

    win_h, win_w = 100, 100
    colln_mnts = []
    for img in colln_imgs:
        img = rgb2yuv(img)
        img_h, img_w = img.shape[0], img.shape[1]
        mean, std_dev, skwn, mnts = [], [], [], []
        for h in range(0, img_h-win_h+1, win_h):
            for w in range(0, img_w-win_w+1, win_w):
                win = img[h:h+win_h, w:w+win_w, :]
                mean.append([np.mean(win[:,:,0]), np.mean(win[:,:,1]), np.mean(win[:,:,2])])
                std_dev.append([np.std(win[:,:,0]), np.std(win[:,:,1]), np.std(win[:,:,2])])
                skwn.append([skew(win[:,:,0], axis=None), skew(win[:,:,1], axis=None), skew(win[:,:,2], axis=None)])
        mnts += [mean, std_dev, skwn]
        colln_mnts.append(mnts)
    colln_mnts = np.array(colln_mnts)
    w,x,y,z = colln_mnts.shape
    colln_mnts = colln_mnts.reshape((w,x*y*z))
    return colln_mnts


def LBP(colln_imgs):
    '''
    Function to compute LBP of list of images
    Input: List of length N, where N is the number of images
    Output: Numpy Array of size (N, M), where N is the number of images and M = 4992
    '''

    win_h, win_w = 100, 100
    colln_lbp = []
    for img in colln_imgs:
        img = rgb2gray(img)
        img_h, img_w = img.shape[0], img.shape[1]
        lbp = []
        for h in range(0, img_h-win_h+1, win_h):
            for w in range(0, img_w-win_w+1, win_w):
                win = img[h:h+win_h, w:w+win_w]
                (hist, _) = np.histogram((local_binary_pattern(win, 24, 8)).ravel(), bins=26, range=(0, 26))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                lbp.append(hist)
        colln_lbp.append(lbp)
    colln_lbp = np.array(colln_lbp)
    x,y,z = colln_lbp.shape
    colln_lbp = colln_lbp.reshape((x,y*z))
    print(colln_lbp.shape)
    exit(0)
    return colln_lbp


def HOG(colln_imgs):
    '''
    Function to compute HOG of list of images
    Input: List of length N, where N is the number of images
    Output: Numpy Array of size (N, M), where N is the number of images and M = 9576
    '''

    colln_hog = []
    for img in colln_imgs:
        img = resize(img, (img.shape[0]//10, img.shape[1]//10), anti_aliasing=True)
        colln_hog.append(hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            block_norm='L2-Hys', feature_vector=True, multichannel=True))
    colln_hog = np.array(colln_hog)
    return colln_hog


def histSIFT(labels, imgkp):

    '''
    Function to calculate histogram of SIFT features of an image
    Input: labels: Cluster membership of every feature
           imgkp: List which contains number of keypoints for an image
    Output: colln_hist: Array of histogram
    '''
    colln_hist = []
    for i in range(0,len(imgkp)-1):
        hist = np.zeros(400)
        label = labels[imgkp[i]:imgkp[i+1]]
        for l in label:
            hist[l] += 1
        norm = sum(hist)
        colln_hist.append([h/norm for h in hist])
    colln_hist = np.array(colln_hist)
    return colln_hist


def kMeansSIFT(colln_sift, imgkp):

    '''
    Function which calculates kmeans of the SIFT descriptors
    Input: colln_sift: Array of SIFT descriptors
           imgkp: List which contains number of keypoints for an image
    Output: colln_hist: Array of histogram
    '''    
    kmeans = KMeans(n_clusters=400)
    kmeans.fit(colln_sift)
    labels = kmeans.labels_
    colln_hist = histSIFT(labels, imgkp)
    '''
    #Deciding the number of clusters using elbow method
    dist = []
    for i in range(50,1001,50):
        print(i)
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(colln_sift)
        dist.append(kmeans.inertia_)
    plt.plot(list(range(50,1001,50)), dist)
    plt.show()
    exit(0)
    '''
    with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/kmeans'+'_'+str(400)+'.joblib', 'wb') as f1:
        joblib.dump(kmeans, f1)
    return colln_hist


def SIFTFeatures(colln_imgs):

    '''
    Function to calculate SIFT features
    Input: colln_imgs: List of images
    Output: colln_sift: Array of SIFT descriptors
            imgkp: List which contains number of keypoints for an image
    '''
    colln_sift, imgkp = [], [0]
    sift_cv = xfeatures2d_SIFT.create()
    for img in colln_imgs:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
        _, sift = sift_cv.detectAndCompute(img, None)
        colln_sift += list(sift)
        imgkp.append(np.array(colln_sift).shape[0])
    colln_sift = np.array(colln_sift)
    return colln_sift, imgkp


def SIFT(colln_imgs):

    '''
    Function which calculates histogram of SIFT features of images
    Input: colln_imgs: List of images
    Output: colln_hist: Array of histogram
    '''
    colln_sift, imgkp = SIFTFeatures(colln_imgs)
    colln_hist = kMeansSIFT(colln_sift, imgkp)
    return colln_hist
  

def SVD(A, k):
    '''
    Function to compute SVD
    Input: A: 2D Numpy Array of size (N, M) where N is the number of images and M is number of features
           k: Number of reduced dimension
    Output: U: Data latent semantic matrix of size (N, k)
            S: List of eigenvalues of length k
            V: Feature latent semantic matrix of size (k, M)
            Ordered in decreasing eigenvalues
    '''

    svd = TruncatedSVD(n_components=k)
    U = svd.fit_transform(A)
    V = svd.components_
    S = svd.singular_values_
    with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/SVD'+'_'+str(k)+'.joblib', 'wb') as f1:
        joblib.dump(svd, f1)
    return U, V, S


def _PCA_(A, k):
    '''
    Function to compute PCA
    Input: A: 2D Numpy Array of size (N, M) where N is the number of images and M is number of features
           k: Number of reduced dimension
    Output: U: Data latent semantic matrix of size (N, k)
            V: Feature latent semantic matrix (principal components) of size (k, M)
            S: List of eigenvalues of length k
            Ordered in decreasing eigenvalues
    '''

    pca = PCA(n_components=k)
    U = pca.fit_transform(A)
    V = pca.components_
    S = pca.singular_values_
    with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/PCA'+'_'+str(k)+'.joblib', 'wb') as f1:
        joblib.dump(pca, f1)
    return U, V, S


def _NMF_(A, k):
    '''
    Function to compute NMF
    Input: A: 2D Numpy Array of size (N, M) where N is the number of images and M is number of features
           k: Number of reduced dimension
    Output: U: Data latent semantic matrix (data in k-dimensional space) of size (N, k)
            V: Feature latent semantic matrix (dictionary/factorization matrix) of size (k, M)
    '''

    nmf = NMF(n_components=k)
    U = nmf.fit_transform(A)
    V = nmf.components_
    with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/NMF'+'_'+str(k)+'.joblib', 'wb') as f1:
        joblib.dump(nmf, f1)
    return U, V, np.array([])


def LDA(A, k):
    '''
    Function to compute LDA
    Input: A: 2D Numpy Array of size (N, M) where N is the number of images and M is number of features
           k: Number of reduced dimension
    Output: U: Data latent semantic matrix (data in k-dimensional space) of size (N, k)
            V: Feature latent semantic matrix of size (k, M)
    '''

    lda = LatentDirichletAllocation(n_components=k)
    U = lda.fit_transform(A)
    V = lda.components_
    with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/LDA'+'_'+str(k)+'.joblib', 'wb') as f1:
        joblib.dump(lda, f1)
    return U, V, np.array([])


def featureModel(imgs, f):
    '''
    Function to calculate features
    Input: imgs: List of images
           f: Label depicting the feature model
    Output: A: Array of features
    '''

    if f=='LBP':
        A = LBP(imgs)
    elif f=='HOG':
        A = HOG(imgs)
    elif f=='SIFT':
        A = SIFT(imgs)
    else:
        A = moments(imgs)
    return A


def dimRed(A, d, k):
    '''
    Function for dimensionality reduction
    Input: A: Array of features
           d: Label depicting the dimensionality reduction technique
           k: Number of reduced dimension
    Output: U: Data latent semantic matrix of size (N, k)
            V: Feature latent semantic matrix (principal components) of size (k, M)
            S: List of eigenvalues of length k
            Ordered in decreasing eigenvalues
    '''

    if d=='SVD':
        U, V, S = SVD(A, k)
    elif d=='PCA':
        U, V, S = _PCA_(A, k)
    elif d=='NMF':
        U, V, S = _NMF_(A, k)
    else:
        U, V, S = LDA(A, k) 
    return U, V, S


def latentSemantics(imgs, f, d, k):
    '''
    Function to compute data latent semantics and feature latent semantics
    Input: imgs: List of images of length N
           f: Feature model
           d: Dimensionality reduction technique
           k: Number of reduced dimension
    Output: U: Data latent semantic matrix of size (N, k)
            V: Feature latent semantic matrix of size (k, M)
            S: List of eigenvalues of length k
            Ordered in decreasing eigenvalues
    '''

    A = featureModel(imgs, f)
    U, V, S = dimRed(A, d, k)
    return U, V, S


def termWeight(U, V, S):
    '''
    Function to compute term-weight pair
    Input: U: Data latent semantic matrix of size (N, k)
           V: Feature latent semantic matrix of size (k, M)
           S: List of eigenvalues of length k
    Output: dataLS: Dictionary of length k, where key corresponds to a data latent semantic 
                    and its corresponding value is a list of tuples of size N
                    where each tuple is a term weight pair ordered in decreasing weight
            featureLS: Dictionary of length k, where key corresponds to a feature latent semantic 
                       and its corresponding value is a list of tuples of size M
                       where each tuple is a term weight pair ordered in decreasing weight
    '''

    dataLS, featureLS = {}, {}

    if S.size:
        U = np.divide(U, S)
    
    for k in range(U.shape[1]):
        ls = U[:,k]
        tw = [(t+1,w) for t,w in enumerate(ls)]
        tw.sort(key=lambda x: x[1], reverse=True)
        dataLS['Latent Semantic'+' #'+str(k+1)] = tw
    
    for k in range(V.shape[0]):
        ls = V[k,:]
        tw = [(t+1,w) for t,w in enumerate(ls)]
        tw.sort(key=lambda x: x[1], reverse=True)
        featureLS['Latent Semantic'+' #'+str(k+1)] = tw
    
    return dataLS, featureLS


def findSimilarImgs(U, m, idx, q):
    '''
    Function to find m most similar images
    Input: U: Data latent semantic matrix of size (N, k)
           m: Number of similar images
           idx: Index of the query image
           q: Latent semantic of the query image
    Output: idxs: Indexes corresponding to the m most similar images 
            scores: Matching score for the m most similar images in decreasing order
    '''

    #Cosine
    dist = [-1]*U.shape[0]
    if not q.size:
        q = U[idx,:]
        for i, f in enumerate(U):
            if i!=idx:
                dist[i] = cosineDist(q, f)
    else:
        q = q[0,:]
        for i, f in enumerate(U):
            dist[i] = cosineDist(q, f)
    idxs = np.argpartition(dist, -m)[-m:]
    scores = [dist[i] for i in idxs]
    z = list(zip(idxs,scores))
    z.sort(key=lambda x: x[1], reverse=True)
    idxs, scores = list(zip(*z))[0], list(zip(*z))[1]

    '''
    #Euclidean
    dist = [-1]*U.shape[0]
    if not q.size:
        q = U[idx,:]
        for i, f in enumerate(U):
            if i!=idx:
                dist[i] = euclideanDist(q, f)
        dist[idx] = max(dist)
        mn, mx = min(dist), max(dist)
        dist = [(1-((x-mn)/(mx-mn)))*100 for x in dist]
    else:
        q = q[0,:]
    idxs = np.argpartition(dist, -m)[-m:]
    scores = [dist[i] for i in idxs]
    z = list(zip(idxs,scores))
    z.sort(key=lambda x: x[1], reverse=True)
    idxs, scores = list(zip(*z))[0], list(zip(*z))[1]
    '''

    '''
    if metric=='cosine':
        for i, f in enumerate(U):
            if i!=idx:
                dist[i] = cosineDist(q, f)
    else:
        for i, f in enumerate(U):
            if i!=idx:
                dist[i] = euclideanDist(q, f)
        if flag:
            dist[idx] = max(dist)
        mn, mx = min(dist), max(dist)
        dist = [(1-((x-mn)/(mx-mn)))*100 for x in dist]
    idxs = np.argpartition(dist, -m)[-m:]
    scores = [dist[i] for i in idxs]
    z = list(zip(idxs,scores))
    z.sort(key=lambda x: x[1], reverse=True)
    idxs, scores = list(zip(*z))[0], list(zip(*z))[1]
    '''
    return idxs, scores


def displaySimImgs(qImg, imgID, simImgs, simImgID, scores):
    """
    Function to display m similar images with name and matching score
    Input: qImg: Query image
           imgID: Name of the query image 
           simImgs: List of m most similar images of length m
           simImgID: List of names of m most similar images of length m
           scores: List of matching scores of m most similar images of length m
    """

    cols = 3
    p = 1
    N = len(simImgID)
    rows = N//cols + 1
    
    plt.figure(figsize=(10, 10))  
    plt.subplot(rows, cols, p)        
    plt.title('Hand_'+imgID+'.jpg   Query Image')
    plt.imshow(qImg)        
    p += 1    
    for i,v in enumerate(simImgID):
        plt.subplot(rows, cols, p)        
        plt.title('Hand_'+v+'.jpg   Score: '+str(scores[i]))
        plt.imshow(simImgs[i])        
        p += 1   
    plt.subplots_adjust(hspace=0.5, wspace=0.1)  
    plt.show()


def classify(U, q, l):
    '''
    Function for classification
    Input: U: Features of images of label l in k reduced dimensions
           q: Feature of query image in k reduced dimension
           l: Label 
    Output: Class
    '''

    #SVM
    svm = OneClassSVM(gamma='auto')
    svm.fit(U)
    if svm.predict(q)==1:
        return l
    else:
        return 'not ' + l
    
    '''
    #Own Approach
    centroid = np.mean(U, axis=0)
    mDist = min([cosineDist(centroid, f) for f in U])
    print(mDist)
    if cosineDist(centroid, q[0,:]) >= mDist:
        return l
    else:
        return 'not '+l
    '''


def binaryImgMetadata(metaData):
    '''
    Function to compute binary Image-Metadata matrix
    Input: metaData: metaData: Array of size (N, 9)
    Output:imgMeta: Array of size (N, 8)
    '''

    imgMetaData = []
    for r in metaData:
        male = female = left = right = dorsal = palmar = wacc = woacc = 0
        if r[2]=='male':
            male = 1
        else:
            female = 1
        if r[4]==1:
            wacc = 1
        else:
            woacc = 1
        if 'dorsal' in r[6]:
            dorsal = 1
        else:
            palmar = 1
        if 'left' in r[6]:
            left = 1
        else:
            right = 1
        imgMetaData.append([male, female, left, right, dorsal, palmar, wacc, woacc])
    return np.array(imgMetaData)
    

def main():

    PATH = '/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Data/Test/'
    imgID = getImgID(PATH)
    imgs = loadImgs(PATH)
    metaData = getMetadata(PATH, imgID)
    
    t = int(sys.argv[1])
    
    #Task 1
    if t==1:
        f, k, d = sys.argv[2], int(sys.argv[3]), sys.argv[4]

        U, V, S = latentSemantics(imgs, f, d, k)
               
        dataLS, featureLS = termWeight(U, V, S)
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'_'+'DataLatentSemantics'+'.json', 'w') as f1:
                f1.write(json.dumps(dataLS))
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'_'+'FeatureLatentSemantics'+'.json', 'w') as f2:
                f2.write(json.dumps(featureLS))

        dic = {'U':U}
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'.pkl', 'wb') as f3:
                pickle.dump(dic, f3)
    
    #Task 2
    elif t==2:
        f, k, d, img, m = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])

        if os.path.isfile('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'.pkl'):
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'.pkl', 'rb') as f1:
                dic = pickle.load(f1)
            U = dic['U']
        else:
            U, _, _ = latentSemantics(imgs, f, d, k)
            dic = {'U':U}
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'.pkl', 'wb') as f2:
                pickle.dump(dic, f2)
        
        idx = imgID.index(img)
        
        idxs, scores = findSimilarImgs(U, m, idx, np.array([]))

        simImgID = [imgID[i] for i in idxs]
        simImgs = np.array([imgs[i] for i in idxs])
        displaySimImgs(imgs[idx], img, simImgs, simImgID, scores)
    
    #Task 3
    elif t==3:
        f, k, d, l = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]

        imgCorrLabel, _ = getImgCorrLabel(imgs, metaData, l)
        
        U, V, S = latentSemantics(imgCorrLabel, f, d, k)
               
        dataLS, featureLS = termWeight(U, V, S)
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'_'+'DataLatentSemantics'+'.json', 'w') as f1:
                f1.write(json.dumps(dataLS))
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'_'+'FeatureLatentSemantics'+'.json', 'w') as f2:
                f2.write(json.dumps(featureLS))

        dic = {'U':U}
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'.pkl', 'wb') as f3:
                pickle.dump(dic, f3)

    #Task 4
    elif t==4:
        f, k, d, l, img, m = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6], int(sys.argv[7])

        imgCorrLabel, idxCorrLabel = getImgCorrLabel(imgs, metaData, l)
        
        if os.path.isfile('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'.pkl'):
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'.pkl', 'rb') as f4:
                dic = pickle.load(f4)
            U = dic['U']
        else:
        
            U, _, _ = latentSemantics(imgCorrLabel, f, d, k) 
            dic = {'U':U}
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'.pkl', 'wb') as f1:
                    pickle.dump(dic, f1)
        
        idx = imgID.index(img)
        
        if idx in idxCorrLabel:
            idxs, scores = findSimilarImgs(U, m, idxCorrLabel.index(idx), np.array([]))

            simImgID = [imgID[idxCorrLabel[i]] for i in idxs]
            simImgs = np.array([imgs[idxCorrLabel[i]] for i in idxs])
            displaySimImgs(imgs[idx], img, simImgs, simImgID, scores)  
        else:
            qImg = [imgs[idx]]

            if f=='SIFT':
                colln_sift, imgkp = SIFTFeatures(qImg)
                with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/kmeans'+'_'+str(400)+'.joblib', 'rb') as f2:
                    kmeans = joblib.load(f2)
                labels = kmeans.predict(colln_sift)
                A = histSIFT(labels, imgkp)
            else:
                A = featureModel(qImg, f)

            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+d+'_'+str(k)+'.joblib', 'rb') as f3:
                model = joblib.load(f3)
            q = model.transform(A)
            
            idxs, scores = findSimilarImgs(U, m, idx, q)

            simImgID = [imgID[idxCorrLabel[i]] for i in idxs]
            simImgs = np.array([imgs[idxCorrLabel[i]] for i in idxs])
            displaySimImgs(imgs[idx], img, simImgs, simImgID, scores)  

    #Task 5
    elif t==5:
        f, k, d, l, img = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]

        if os.path.isfile('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'.pkl'):
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'.pkl', 'rb') as f4:
                dic = pickle.load(f4)
            U = dic['U']
        else:
            U, _, _ = latentSemantics(imgCorrLabel, f, d, k)
                
            dic = {'U':U}
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'.pkl', 'wb') as f1:
                    pickle.dump(dic, f1)
        
        qImg = imread_collection(PATH+'Hand_'+img+'.jpg')
        
        if f=='SIFT':
            colln_sift, imgkp = SIFTFeatures(qImg)
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/kmeans'+'_'+str(400)+'.joblib', 'rb') as f2:
                kmeans = joblib.load(f2)
            labels = kmeans.predict(colln_sift)
            A = histSIFT(labels, imgkp)
        else:
            A = featureModel(qImg, f)

        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+d+'_'+str(k)+'.joblib', 'rb') as f3:
            model = joblib.load(f3)
        q = model.transform(A)
        
        print(classify(U, q, l))
    
    #task 6
    elif t==6:
        import itertools
        img = sys.argv[2]
        metric = 'euclidean'
        k = 8
        idx = imgID.index(img)

        feature = ['LBP', 'HOG', 'SIFT', 'MON']
        dim = ['PCA', 'SVD', 'NMF', 'LDA']
        a = list(itertools.product(feature, dim))
        simidex = []
        for i in a:
            f = i[0]
            d = i[1]
            if os.path.isfile('/home/yhc/CSE515/Features/' + f + '_' + d + '_' + str(k) + '.pkl'):
                with open('/home/yhc/CSE515/Features/' + f + '_' + d + '_' + str(k) + '.pkl', 'rb') as f1:
                    dic = pickle.load(f1)
                U = dic['U']
            else:
                U, V = latentSemantics(imgs, f, d, k)
                dic = {'U': U, 'V': V}
                with open('/home/yhc/CSE515/Features/' + f + '_' + d + '_' + str(k) + '.pkl', 'wb') as f2:
                    pickle.dump(dic, f2)
            idxs, _ = findSimilarImgs(U, 6, idx, np.array([]), metric, True)
            simidex.extend(idxs)
        sim = Counter(simidex).most_common(3)
        score = [i[1] for i in sim]
        simImgID = [imgID[i[0]] for i in sim]
        simImgs = np.array([imgs[i[0]] for i in sim])
        displaySimImgs(imgs[idx], img, simImgs, simImgID, score)
        
    #Task 8
    elif t==8:
        k = int(sys.argv[2])

        imgMetaData = binaryImgMetadata(metaData)

        U, V, S = _NMF_(imgMetaData, k)
        
        imageLS, metaDataLS = termWeight(U, V, S)
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+'Image_MetaData'+'_'+'NMF'+'_'+str(k)+'_'+'ImageLatentSemantics'+'.json', 'w') as f1:
                f1.write(json.dumps(imageLS))
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+'Image_MetaData'+'_'+'NMF'+'_'+str(k)+'_'+'MetaDataLatentSemantics'+'.json', 'w') as f2:
                f2.write(json.dumps(metaDataLS))

        
if __name__ == "__main__":

    main()
