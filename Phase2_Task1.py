import numpy as np
import sys
import math
import pickle
from scipy.stats import skew
from skimage.io import imread_collection, imread, imshow, show, imshow_collection
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2yuv, rgb2gray
from skimage.transform import resize
from sklearn.decomposition import NMF, LatentDirichletAllocation, PCA
from cv2 import xfeatures2d_SIFT
import cv2
from scipy.sparse.linalg import svds, eigs


def loadImgs(PATH):
    '''
    Function to load and read the JPG images in the input path
    To read images of different format, change '.jpg' to the required format
    Input: PATH to the folder containing images
    Output: List of size N, where N is the number of images in the folder
    '''

    return(imread_collection(PATH+'*.jpg'))


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
    Output: A real value in range [0,1]
    '''

    dot = np.dot(f1, f2)
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    return 1 - (dot / (norm1 * norm2))


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


def termWeight(M, l):
    '''
    Function to compute term-weight pair
    Input: M: 2D Numpy Array of size (N, k) (in case of l='data') or (k, M) (in case of l='feature'), depending upon the label
           l: Label indicating whether M is data or feature latent semantic matrix. Takes value as 'data' or 'feature'
    Output: P: Dictionary of length k where each value is a list of tuples of size N (in case of l='data') or M (in case of l='feature') 
               where each tuple is a term weight pair ordered in decreasing weight
    '''

    dic = {}
    if l=='data':
        for k in range(M.shape[1]):
            ls = M[:,k]
            tw = [(t+1,w) for t,w in enumerate(ls)]
            tw.sort(key=lambda x: x[1], reverse=True)
            dic['Data Latent Semantic'+' #'+str(k+1)] = tw
    else:
        for k in range(M.shape[0]):
            ls = M[k,:]
            tw = [(t+1,w) for t,w in enumerate(ls)]
            tw.sort(key=lambda x: x[1], reverse=True)
            dic['Feature Latent Semantic'+' #'+str(k+1)] = tw
    return dic


def main():

    PATH = '/home/pu/Desktop/CSE515/Project/Phase1/Priyansh_Phase1/Data/Test/'
    f, k, d = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    
    if f=='LBP':
        A = LBP(loadImgs(PATH))
    elif f=='HOG':
        A = HOG(loadImgs(PATH))
    elif f=='SIFT':
        A = SIFT(loadImgs(PATH))
    else:
        A = moments(loadImgs(PATH))
    
    if d=='SVD':
        U, _, V = SVD(A, k)
        dataLS = termWeight(U, 'data')
        featureLS = termWeight(V, 'feature')
    elif d=='PCA':
        U, V, _, _ = _PCA_(A, k)
        dataLS = termWeight(U, 'data')
        featureLS = termWeight(V, 'feature')
    elif d=='NMF':
        U, V = _NMF_(A, k)
        dataLS = termWeight(U, 'data')
        featureLS = termWeight(V, 'feature')
    else:
        U, V = LDA(A, k)
        dataLS = termWeight(U, 'data')
        featureLS = termWeight(V, 'feature')


if __name__ == "__main__":

    main()