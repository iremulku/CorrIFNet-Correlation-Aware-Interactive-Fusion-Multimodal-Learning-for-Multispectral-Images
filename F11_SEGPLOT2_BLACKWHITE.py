from __future__ import print_function
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def segplot(pathm, lim, image, predmask, grmask, trMeanR, trMeanG, trMeanB, indx):    
       
    image[:,:,0] = image[:,:,0] + trMeanR
    image[:,:,1] = image[:,:,1] + trMeanG
    image[:,:,2] = image[:,:,2] + trMeanB
    image = (image-np.min(image))/(np.max(image)-np.min(image))
    rgbname="rgb_{}.png".format(indx)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite(os.path.join(pathm, rgbname),gray*255)
    
    c = np.zeros_like(grmask)
    c[:, 1::3] = 1
    

    outname="segmentation_image_{}.png".format(indx)
    out=predmask*0.7+c*grmask*0.3
    cv2.imwrite(os.path.join(pathm, outname),out*255)
     
    #plt.imsave(os.path.join(pathm, outname),predmask*0.7+c*grmask*0.3,cmap="gray")


    maskname="mask_{}.png".format(indx)
    cv2.imwrite(os.path.join(pathm, maskname),grmask*255) 
    # plt.imsave(os.path.join(pathm, 'test_image.png'),image)
    # plt.imsave(os.path.join(pathm, 'test_image_R.png'),image[:,:,0],cmap="gray")
    # plt.imsave(os.path.join(pathm, 'test_image_G.png'),image[:,:,1],cmap="gray")
    # plt.imsave(os.path.join(pathm, 'test_image_B.png'),image[:,:,2],cmap="gray")
    # plt.imsave(os.path.join(pathm, 'test_pred_mask.png'),np.squeeze(predmask))
    # plt.imsave(os.path.join(pathm, 'ground_truth_mask.png'),np.squeeze(grmask))
                







