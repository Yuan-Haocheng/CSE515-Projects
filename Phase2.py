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
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from cv2 import xfeatures2d_SIFT
import cv2
from scipy.sparse.linalg import svds, eigs
import pandas


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
    '''

    imgCorrLabel = []
    imgs = np.array(imgs)
    if l=='left' or l=='right' or l=='dorsal' or l=='palmar':
        for i,v in enumerate(list(metaData[:,6])):
            if l in v:
                imgCorrLabel.append(imgs[i])
    elif l=='male' or l=='female':
        for i,v in enumerate(list(metaData[:,2])):
            if l==v:
                imgCorrLabel.append(imgs[i])
    elif l=='wacc':
        for i,v in enumerate(list(metaData[:,4])):
            if v==1:
                imgCorrLabel.append(imgs[i])
    else:
        for i,v in enumerate(list(metaData[:,4])):
            if v==0:
                imgCorrLabel.append(imgs[i])   
    return imgCorrLabel             


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
                (hist, _) = np.histogram((local_binary_pattern(win, 24, 8)).ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
                lbp.append(hist)
        colln_lbp.append(lbp)
    colln_lbp = np.array(colln_lbp)
    x,y,z = colln_lbp.shape
    colln_lbp = colln_lbp.reshape((x,y*z))
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


def SIFT(colln_imgs):
    '''
    Function to compute SIFT of list of images
    Input: List of length N, where N is the number of images
    Output: Numpy Array of size (N, M), where N is the number of images and M = 128
    '''

    colln_sift = []
    sift_cv = xfeatures2d_SIFT.create()
    for img in colln_imgs:
        img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
        _, sift = sift_cv.detectAndCompute(img, None)
        sift = np.array(sift)
        colln_sift.append(np.mean(sift,axis=0))
    return np.array(colln_sift)
  

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

    U, S, V = svds(A, k=k)
    U = U[:, ::-1]
    S = S[::-1]
    V = V[::-1, :]
    return U, S, V


def _PCA_(A, k):
    '''
    Function to compute PCA
    Input: A: 2D Numpy Array of size (N, M) where N is the number of images and M is number of features
           k: Number of reduced dimension
    Output: U: Data latent semantic matrix (data in k-dimensional space) of size (N, k)
            V: Feature latent semantic matrix (principal components) of size (k, M)
            S: List of eigenvalues of length k
            D: List of percentage of variance accounted by each principal component
            Ordered in decreasing eigenvalues
    '''

    pca = PCA(n_components=k)
    U = pca.fit_transform(A)
    V = pca.components_
    S = pca.explained_variance_
    D = pca.explained_variance_ratio_ 
    return U, V, S, D


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
    return U, V    


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
    return U, V  


def latentSemantics(imgs, f, d, k):
    '''
    Function to compute data latent semantics and feature latent semantics
    Input: imgs: List of images of length N
           f: Feature model
           d: Dimensionality reduction technique
           k: Number of reduced dimension
    Output: U: Data latent semantic matrix of size (N, k)
            V: Feature latent semantic matrix of size (k, M)
    '''

    if f=='LBP':
        A = LBP(imgs)
    elif f=='HOG':
        A = HOG(imgs)
    elif f=='SIFT':
        A = SIFT(imgs)
    else:
        A = moments(imgs)
        
    if d=='SVD':
        U, _, V = SVD(A, k)
    elif d=='PCA':
        U, V, _, _ = _PCA_(A, k)
    elif d=='NMF':
        U, V = _NMF_(A, k)
    else:
        U, V = LDA(A, k) 

    return U, V


def termWeight(U, V):
    '''
    Function to compute term-weight pair
    Input: U: Data latent semantic matrix of size (N, k)
           V: Feature latent semantic matrix of size (k, M)
    Output: dataLS: Dictionary of length k, where key corresponds to a data latent semantic 
                    and its corresponding value is a list of tuples of size N
                    where each tuple is a term weight pair ordered in decreasing weight
            featureLS: Dictionary of length k, where key corresponds to a feature latent semantic 
                       and its corresponding value is a list of tuples of size M
                       where each tuple is a term weight pair ordered in decreasing weight
    '''

    dataLS, featureLS = {}, {}
    
    for k in range(U.shape[1]):
        ls = U[:,k]
        tw = [(t+1,w) for t,w in enumerate(ls)]
        tw.sort(key=lambda x: x[1], reverse=True)
        dataLS['Data Latent Semantic'+' #'+str(k+1)] = tw
    
    for k in range(V.shape[0]):
        ls = V[k,:]
        tw = [(t+1,w) for t,w in enumerate(ls)]
        tw.sort(key=lambda x: x[1], reverse=True)
        featureLS['Feature Latent Semantic'+' #'+str(k+1)] = tw
    
    return dataLS, featureLS


def findSimilarImgs(U, m, idx, metric):
    '''
    Function to find m most similar images
    Input: U: Data latent semantic matrix of size (N, k)
           m: Number of similar images
           idx: Index of the query image
           metric: Evaluation metric
    Output: idxs: Indexes corresponding to the m most similar images 
            scores: Matching score for the m most similar images in decreasing order
    '''

    dist = []
    q = U[idx,:]
    dist = [-1]*U.shape[0]
    if metric=='cosine':
        for i, f in enumerate(U):
            if i!=idx:
                dist[i] = cosineDist(q, f)
    else:
        for i, f in enumerate(U):
            if i!=idx:
                dist[i] = euclideanDist(q, f)
        mn, mx = min(dist), max(dist)
        dist = [(1-(x-mn)/(mx-mn))*100 for x in dist]
    idxs = np.argpartition(dist, -m)[-m:]
    scores = [dist[i] for i in idxs]
    z = list(zip(idxs,scores))
    z.sort(key=lambda x: x[1], reverse=True)
    idxs, scores = list(zip(*z))[0], list(zip(*z))[1]
    return idxs, scores


def displaySimImgs(simImgs, simImgID, scores):
    """
    Function to display m similar images with name and matching score
    Input: simImgs: List of m most similar images of length m
           simImgID: List of names of m most similar images of length m
           scores: List of matching scores of m most similar images of length m
    """

    cols = 3
    p = 1
    N = len(simImgID)
    rows = N//cols + 1
    
    plt.figure(figsize=(10, 10))       
    for i,v in enumerate(simImgID):
        plt.subplot(rows, cols, p)        
        plt.title('Hand_'+v+'.jpg   Score: '+str(scores[i]))
        plt.imshow(simImgs[i])        
        p += 1   
    plt.subplots_adjust(hspace=0.5, wspace=0.1)  
    plt.show()


def main():

    PATH = '/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Data/Test/'
    imgID = getImgID(PATH)
    imgs = loadImgs(PATH)
    metaData = getMetadata(PATH, imgID)
    print(metaData)
    
    t = int(sys.argv[1])
    
    #Task 1
    if t==1:
        f, k, d = sys.argv[2], int(sys.argv[3]), sys.argv[4]

        U, V = latentSemantics(imgs, f, d, k)
               
        dataLS, featureLS = termWeight(U, V)

        dic = {'U':U, 'V':V}
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'.pkl', 'wb') as f:
                pickle.dump(dic, f)
    
    #Task 2
    elif t==2:
        f, k, d, img, m = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6])

        if os.path.isfile('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'.pkl'):
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'.pkl', 'rb') as f:
                dic = pickle.load(f)
            U = dic['U']
        else:
            U, V = latentSemantics(imgs, f, d, k)
            dic = {'U':U, 'V':V}
            with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+str(k)+'.pkl', 'wb') as f:
                pickle.dump(dic, f)
        
        idx = imgID.index(img)
        
        idxs, scores = findSimilarImgs(U, m, idx, 'cosine')

        simImgID = [imgID[i] for i in idxs]
        simImgs = np.array([imgs[i] for i in idxs])
        displaySimImgs(simImgs, simImgID, scores)
    
    #Task 3
    if t==3:
        f, k, d, l = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]

        imgCorrLabel = getImgCorrLabel(imgs, metaData, l)
        
        U, V = latentSemantics(imgCorrLabel, f, d, k)
               
        dic = {'U':U, 'V':V}
        with open('/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Features/'+f+'_'+d+'_'+l+'_'+str(k)+'.pkl', 'wb') as f:
                pickle.dump(dic, f)


if __name__ == "__main__":

    main()
