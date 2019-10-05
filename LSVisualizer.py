import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def dataLSVisualize(eigen_vectors, weights=None, colln_imgs): # like (5, 44)
    r"""
    Visualizer which shows a ranked list of image thumbnails along with their scores, for each latent semantics.
    --------
    Inputs:
        1. eigen_vectors: A numpy 2-d array of data latent semantics, in which rows = k, cols = number of images.

        2. weights: A numpy 1-d array with length equals to rows of eigen_vectors. Default by None.

        3. colln_imgs: Collection of images, with each image comprised of RGB arrays. len(colln_imgs)=img_num
    """
    
    # print(eigen_vectors) #for debug
    # print(weights)
    # print(colln_imgs)

    imgs = []
    for img in colln_imgs:
        imgs.append(img)
    imgs = np.asarray(imgs)
    # print(imgs.shape) #(44, 1200, 1600, 3)
    for i in range(eigen_vectors.shape[0]):
        vec = eigen_vectors[i, :]
        # print(str(vec.dot(vec.T))) # almost 1, true
        vec_indices = np.argsort(-vec) # descending
        # print(vec_indices)
        print("For the %dth latent semantic" % (i+1))
        vec_len = len(vec)
        row = vec_len / 5 + 1
        col = 5
        plt.figure(num='%dth latent semantic'%(i+1), figsize=(25, 15))
        plt.suptitle("Ranked image list for %dth latent semantic"%(i+1))
        for j in range(vec_len):
            plt.subplot(row, col, j+1)
            plt.title("%dth image with score: %.2f" % (j+1, vec[vec_indices[j]]))#TODO
            plt.imshow(imgs[vec_indices[j]])
            plt.axis('off')
        plt.show()


def featureLSVisualize(eigen_vectors, weights=None, colln_imgs, img_dataset): # like (5, 1728)
    r"""
    Visualizer which, for each latent semantics, selects the image with the "least dot product"  to that latent semantics and visualizes that image as the visual-placeholder for that latent semantics.
    --------
    Inputs:
        1. eigen_vectors: A numpy 2-d array, in which rows = k, cols = dimension of features.

        2. weights: A numpy 1-d array with length equals to rows of eigen_vectors. Default by None.

        3. colln_imgs: Collection of images, with each image comprised of RGB arrays. len(colln_imgs)=img_num

        4. img_dataset: Object-feature matrix for colln_imgs. Shape=(img_num, feature_num)

    """

    # print(len(colln_imgs))
    
    min_semantic = None
    min_img = None
    latents_len = eigen_vectors.shape[0]
    
    plt.figure(num="Visual-placeholders for %d latent semantic." % latents_len, figsize=(25, 15))
    rows = latents_len / 5 + 1
    cols = 5
    for i in range(latents_len):
        vec = eigen_vectors[i]
        min_dot = 1000000
        for j in range(img_dataset.shape[0]):
            new_dot = img_dataset[j].dot(vec.T)
            if new_dot < min_dot:
                min_dot = new_dot
                min_semantic = vec
                min_img = colln_imgs[j]
        print("The %dth latent semantic: " % (i+1))
        print(min_semantic)
        plt.subplot(rows, cols, i+1)
        plt.title("%dth latent semantic" % (i+1))
        '''
            item = ""
            for num in range(len(min_semantic)):
                item += ("%f ") % min_semantic[num]
                if num % 5 == 0:
                    print(item)
                    item = ""
            print("---------------------------")
        '''
        '''
            xlabel = '[' 
            for num in range(latents_len-1):
                xlabel += ("%.2f,\n" % min_semantic[num])
            xlabel += ("%.2f]" % min_semantic[latents_len-1])
            ax.set_xlabel(xlabel)
        '''
        
        plt.imshow(min_img)
        plt.axis('off')
    
    plt.show()