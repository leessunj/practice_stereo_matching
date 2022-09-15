# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import cv2


def matching_for(im,im2,index=16):
    h, w = im.shape  # 288 384
    d = []
    for i in range(h):
        for j in range(w):
            candidates = [255]
            for k in range(0, index):
                if j+k<0:
                    continue
                elif j + k >= w:
                    break
                candidates.append(abs(int(im[i][j + k]) - int(im2[i][j]))) #AD
                #candidates.append((int(im[i][j + k]) - int(im2[i][j]))**2) #SD
            d.append(candidates.index(min(candidates)))
    disparity = np.array(d)
    disparity.shape = (h, w)
    return disparity

def matching_bm(im,im2):
    stereo=cv2.StereoBM_create(numDisparities=16, blockSize=21)
    print(type(stereo))
    disparity=stereo.compute(im,im2)
    return disparity

def matching_block(im,im2,block_size=3,index=12):
    h, w = im.shape  # 288 384
    d=[]
    for i in range(block_size,h):
        for j in range(block_size,w):
            cd=[]
            for k in range(1,index):
                sad=0
                for x in range(j-block_size,j):
                    for y in range(i-block_size,i):
                        if x+k>=w:
                            break
                        sad+=abs(int(im2[y][x])-int(im[y][x+k]))
                cd.append(sad)
                #print(int(im[i-block_size:i,j-block_size:j]))
                #print(im2[i-block_size:i,j-block_size+k:j+k])
                #print((im[i-block_size:i,j-block_size:j]-im2[i-block_size:i,j-block_size+k:j+k]))
            d.append(cd.index(min(cd)))
    disparity = np.array(d)
    disparity.shape = (h-block_size, w-block_size)
    return disparity

if __name__ == '__main__':
    im = cv2.imread('tsukuba/scene1.row3.col1.ppm',0) #left
    im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm',0) #right
    optimal_result=cv2.imread('tsukuba/truedisp.row3.col3.pgm')
    #matching_block(im,im2)
    disparity=matching_block(im, im2)
    cv2.imshow("image1",im)
    cv2.imshow("2", im2)
    plt.imshow(disparity,'gray')
    plt.show()


