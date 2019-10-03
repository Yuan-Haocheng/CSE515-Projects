import numpy as np
from scipy.sparse.linalg import svds
from phase1_libs import moments, load_imgs
from LSVisualizer import dataLSVisualize as dv
from LSVisualizer import featureLSVisualize as fv
import utilities as util

def SVD(input_matrix, k=5):
    u, s, vh = svds(A=input_matrix, k=k, which='LM')
    return u, s, vh

def dataPreprocess(raw_data): # (img_nums, i_th moment, position, YUV)
    img_dataset = []
    for i in range(raw_data.shape[0]):
        img = raw_data[i].flatten()
        img_dataset.append(img)
    img_dataset = np.asarray(img_dataset)
    return img_dataset # (44, 1728)
    
def main(debug_mode=False):
    if debug_mode:
        k = 5
        input_path = 'testset/*'
        colln_imgs = load_imgs(input_path)
        # print(type(colln_imgs)) # <class 'skimage.io.collection.ImageCollection'>
        imgs_moments = np.load("testnpy/10imgs.npy")
        print("The shape of imgs_moments : %s" % str(imgs_moments.shape)) # (44, 3, 192, 3), (img_nums, i_th moment, position, YUV)
        img_dataset = dataPreprocess(imgs_moments) # (44, 1728)
        u, s, vh = SVD(input_matrix=img_dataset, k=k) # Ascending among the top K
        # print(u.shape) # (44, 5)
        # print(s) # [92.76524942 105.2338097  128.5104521  135.10389759 195.24833493]
        # print(vh.shape) # (5, 1728)

        # Below code snippet shows that eigenvectors and values are correspondingly ascending among the top K
        # test = (img_dataset.dot(img_dataset.T)).dot(u[:, 0])
        # print(test)
        # test = (img_dataset.T.dot(img_dataset)).dot(vh[0, :])
        # print(test)
        # test = s[0]**2 * vh[0, :]
        # print(test) 
    else:
        k = int(input("Please input K:\n"))
        input_path = input("Please enter the path of images:\n")
        print("Please wait for a second, features are being generated...")
        colln_imgs = load_imgs(input_path)
        imgs_moments = np.asarray(moments(colln_imgs))
        img_num = len(colln_imgs)
        np.save(file="testnpy/%dimgs.npy" % img_num, arr=imgs_moments)
        img_dataset = dataPreprocess(imgs_moments) # (44, 1728)
        u, s, vh = SVD(input_matrix=img_dataset, k=k) # Ascending among the top K

    # Ascending to desending
    u = np.flip(u, axis=1) # (44, 5)
    s = np.flip(s) # (5, )
    vh = np.flip(vh, axis=0) # (5, 1728)

    # print(s) debug
    # test = (img_dataset.dot(img_dataset.T)).dot(u[:, 0])
    # print(test)
    # test = s[0]**2 * u[:, 0]
    # print(test) 
    # test = u[:, 0].T.dot(u[:, 0])
    # print(test)
    
    choice = input("Do you want to visualize latent semantics (Data)? [Y/N]\n")
    if choice == 'Y' or choice == 'y':
        print("Visualization for data latent semantics.")
        dv(u.T, s, colln_imgs)
    choice = input("Do you want to visualize latent semantics (Feature)? [Y/N]\n")
    if choice == 'Y' or choice == 'y':
        print("Visualization for feature latent semantics.")
        fv(vh, s, colln_imgs, img_dataset)


if __name__ == "__main__":
    choice = input("Debug/Release? [D/R]\n")
    if choice == 'D' or choice == 'd':
        debug_mode = True
    else:
        debug_mode = False
    main(debug_mode=debug_mode)