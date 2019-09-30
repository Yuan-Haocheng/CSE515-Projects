import numpy as np
import sys
import pickle
from scipy.stats import skew
from skimage.io import imread_collection, imread, imshow, show, imshow_collection
from skimage.feature import local_binary_pattern, hog
from skimage.color import rgb2yuv, rgb2gray
from skimage.transform import resize
from cv2 import xfeatures2d_SIFT
import cv2


#Function to load and read the images in the input path
def load_imgs(PATH):

    #Read and return a collection of images in the input path
    return(imread_collection(PATH))


def moments(colln_imgs):

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
    return colln_mnts


#Function to calculate LBP features of an image
def LBP(colln_imgs):

    win_h, win_w = 100, 100
    colln_lbp = []
    for img in colln_imgs:
        img = rgb2gray(img)
        img_h, img_w = img.shape[0], img.shape[1]
        lbp = []
        for h in range(0, img_h-win_h+1, win_h):
            for w in range(0, img_w-win_w+1, win_w):
                win = img[h:h+win_h, w:w+win_w]
                lbp.append(local_binary_pattern(win, 8, 1))
                (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-7)
        colln_lbp.append(hist)
    return colln_lbp


#Function to calculate HOG features of an image
def HOG(colln_imgs):

    colln_hog = []
    #Repeat for every image
    for img in colln_imgs:
        #Down sampling the image
        img = resize(img, (img.shape[0]//10, img.shape[1]//10), anti_aliasing=True)
        #Computing HOG features and appending it onto the list
        colln_hog.append(hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            block_norm='L2-Hys', feature_vector=True, multichannel=True))
    return colln_hog


#Function to calculate SIFT features of an image
def SIFT(colln_imgs):

    colln_sift = []
    #Constructing a SIFT object
    sift_cv = xfeatures2d_SIFT.create()
    #Repeat for every image
    for img in colln_imgs:
        #Converting the color image into gray-scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #Computing SIFT descriptors 
        _, sift = sift_cv.detectAndCompute(img, None)
        #Appending the SIFT descriptors onto the list
        colln_sift.append(sift)
    return colln_sift


#Function to calculate Cosine Distance for HOG features
def cosineHOG(f1, f2):

    #Calculating dot product between two image features
    dot = np.dot(f1, f2)
    #Computing norm for the first image features
    norm1 = np.linalg.norm(f1)
    #Computing norm for the second image features
    norm2 = np.linalg.norm(f2)
    #Calculating and returning the cosine distance
    return 1 - (dot / (norm1 * norm2))
    

#Function to calculate Cosine Distance for SIFT features
def cosineSIFT(f1, f2):

    cos = 0
    #Repeat for every descriptor of image 1
    for f in f1:
        #Repeat for every descriptor of image 2
        for fc in f2:
            #Calculating dot product between two descriptors
            dot = np.dot(f, fc)
            #Computing norm for the first descriptor
            norm1 = np.linalg.norm(f)
            #Computing norm for the second descriptor
            norm2 = np.linalg.norm(fc)
            #Calculating and summing up the cosine distance
            cos += 1 - (dot / (norm1 * norm2))
    #Returning the mean of the cosine distance
    return cos/(len(f1)*len(f2))


def main():

    print('Main Function')


if __name__ == "__main__":

    main()