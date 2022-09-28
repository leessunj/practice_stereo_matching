# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math


def matching_for(im, im2, index=16):
    h, w = im.shape  # 288 384
    d = []
    for i in range(h):
        for j in range(w):
            candidates = [255]
            for k in range(0, index):
                if j + k < 0:
                    continue
                elif j + k >= w:
                    break
                # candidates.append(abs(int(im[i][j + k]) - int(im2[i][j]))) #AD
                candidates.append((int(im[i][j + k]) - int(im2[i][j])) ** 2)  # SD
            d.append(candidates.index(min(candidates)))
    disparity = np.array(d)
    disparity.shape = (h, w)
    return disparity


def matching_bm(im, im2):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
    print(type(stereo))
    disparity = stereo.compute(im, im2)
    return disparity


def matching_block(im, im2, block_size=3, index=12):
    h, w = im.shape  # 288 384
    d = []
    for i in range(block_size, h):
        for j in range(block_size, w):
            cd = []
            for k in range(1, index):
                sad = 0
                for x in range(j - block_size, j):
                    for y in range(i - block_size, i):
                        if x + k >= w:
                            break
                        sad += abs(int(im2[y][x]) - int(im[y][x + k]))
                        # sad += (int(im2[y][x]) - int(im[y][x + k]))**2 #SSD
                cd.append(sad)
                # print(int(im[i-block_size:i,j-block_size:j]))
                # print(im2[i-block_size:i,j-block_size+k:j+k])
                # print((im[i-block_size:i,j-block_size:j]-im2[i-block_size:i,j-block_size+k:j+k]))
            d.append(cd.index(min(cd)))
    disparity = np.array(d)
    disparity.shape = (h - block_size, w - block_size)
    return disparity


w_dp={}

def cpq(im, p, q):
    if p == q:
        return 0
    #print(p,q)
    ld = int(im[p[0]][p[1]][0]) - int(im[q[0]][q[1]][0])
    ad = int(im[p[0]][p[1]][1]) - int(im[q[0]][q[1]][1])
    bd = int(im[p[0]][p[1]][2]) - int(im[q[0]][q[1]][2])
    cdiff = math.sqrt(ld ** 2 + ad ** 2 + bd ** 2)
    return cdiff

def gpq(p,q):
    return (p[0]-q[0])**2+(p[1]-q[1])**2

def w(im,p,q,rc=7,rp=36):
    global w_dp
    if (p,q) in w_dp:
        return w_dp[p,q]
    w=math.exp(-(cpq(im,p,q)/rc + gpq(p,q)/rp))*5
    w_dp[(p,q)]=w
    #print(p,q,w)
    return w

def dis_a(im1,p1,q1,im2,p2,q2,im1_g,im2_g):
    e0=abs(int(im1_g[q1[0]][q1[1]]) - int(im2_g[q2[0]][q2[1]]))
    return w(im1,p1,q1)*w(im2,p2,q2)*e0

def dis_b(im1,p1,q1,im2,p2,q2):
    return w(im1,p1,q1)*w(im2,p2,q2)

def matching_adaptive_weight(im,im2,block_size=3,max_d=12):
    im1_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2LAB)
    h, w, l = im.shape  # 288 384 3
    print("this is",h,w,l)
    #cv2.imshow("im",im)
    #cv2.imshow('gray',im1_g)
    #cv2.waitKey(0)
    d = []
    for i in range(block_size-1, h):#(block_size, h):
        for j in range(block_size-1, w):
            cd = []
            for k in range(1, max_d):
                sad = 0
                ct_ref=(i-block_size//2,j-block_size//2)
                ct_tar = (i -block_size//2, (j+k) -block_size//2)
                if ct_tar[1]>=w:
                    break
                a=0
                b=0
                for x in range(j+1 - block_size, j):
                    for y in range(i+1 - block_size, i):
                        if x + k >= w:
                            break
                        a+=dis_a(im2,ct_tar,(y,x+k),im,ct_ref,(y,x),im2_g,im1_g)
                        b+=dis_b(im2,ct_tar,(y,x+k),im,ct_ref,(y,x))

                        #sad += abs(int(im2[y][x]) - int(im[y][x + k]))
                b+=1e-10
                cd.append(a/b)
            d.append(cd.index(min(cd)))
        print("{}th line finished".format(i))

    disparity = np.array(d)
    disparity.shape = (h - block_size+1, w - block_size+1)
    return disparity

if __name__ == '__main__':
    im = cv2.imread('tsukuba/scene1.row3.col1.ppm')  # left
    im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm')  # right
    optimal_result = cv2.imread('tsukuba/truedisp.row3.col3.pgm')


    # disparity=matching_bm(im, im2)
    disparity=matching_adaptive_weight(im2, im,block_size=3)
    cv2.imshow("image1", im)
    cv2.imshow("2", im2)
    plt.imshow(disparity,'gray')
    plt.show()
