# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import cv2


def matching_for(im,im2):
    h, w = im.shape  # 288 384
    d = []
    index=int(8/2)
    for i in range(h):
        for j in range(w):
            candidates = [255]
            for k in range(-index, index):
                if j+k<0:
                    continue
                elif j + k >= w:
                    break
                candidates.append(abs(int(im[i][j + k]) - int(im2[i][j])))
            d.append(candidates.index(min(candidates)))
    disparity = np.array(d)
    disparity.shape = (h, w)
    return disparity

def matching_bm(im,im2):
    stereo=cv2.StereoBM_create(numDisparities=16, blockSize=21)
    print(type(stereo))
    disparity=stereo.compute(im,im2)
    return disparity

if __name__ == '__main__':
    im = cv2.imread('tsukuba/scene1.row3.col1.ppm',0) #left
    im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm',0) #right
    optimal_result=cv2.imread('tsukuba/truedisp.row3.col3.pgm')
    #im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    h, w = im.shape  # 288 384

    disparity=matching_for(im,im2)
    cv2.imshow("image1",im)
    cv2.imshow("2", im2)
    plt.imshow(disparity,'gray')
    plt.show()


