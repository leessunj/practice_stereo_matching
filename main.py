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
    disparity=disparity/np.max(disparity)
    disparity.shape = (h, w)

    return disparity

def matching_bm(im, im2):
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=21)
    print(type(stereo))
    disparity = stereo.compute(im, im2)
    return disparity

def matching_block(im, im2, block_size=3, index=12): #left+d==right
    if len(im.shape)>2:
        im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        im2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    h, w = im.shape  # 288 384
    d = []
    for i in range(block_size-1, h):
        for j in range(block_size-1, w):
            cd = []
            for k in range(1, index):
                sad = 0
                for x in range(j - block_size+1, j):
                    for y in range(i - block_size+1, i):
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
    disparity=disparity/np.max(disparity)
    disparity.shape = (h - block_size+1, w - block_size+1)
    return disparity

def matching_block_rl(im, im2, block_size=3, index=12): #left==right+d
    if len(im.shape)>2:
        im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        im2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    h, w = im.shape  # 288 384
    d = []
    for i in range(block_size-1, h):
        for j in range(block_size-1, w):
            cd = []
            for k in range(1, index):
                sad = 0
                for x in range(j - block_size+1, j):
                    for y in range(i - block_size+1, i):
                        if x - k < 0:
                            break
                        sad += abs(int(im2[y][x]) - int(im[y][x - k]))
                        # sad += (int(im2[y][x]) - int(im[y][x + k]))**2 #SSD
                cd.append(sad)
                # print(int(im[i-block_size:i,j-block_size:j]))
                # print(im2[i-block_size:i,j-block_size+k:j+k])
                # print((im[i-block_size:i,j-block_size:j]-im2[i-block_size:i,j-block_size+k:j+k]))
            d.append(cd.index(min(cd)))
    disparity = np.array(d)
    disparity=disparity/np.max(disparity)
    disparity.shape = (h - block_size+1, w - block_size+1)
    return disparity

def matching_adaptive_weight(im,im2,block_size=3,max_d=12):
    im1_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im2_g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2LAB)
    h, w = im1_g.shape  # 288 384 3

    w_dp = {}

    def cpq(im, p, q):
        if p == q:
            return 0
        # print(p,q)
        ld = int(im[p[0]][p[1]][0]) - int(im[q[0]][q[1]][0])
        ad = int(im[p[0]][p[1]][1]) - int(im[q[0]][q[1]][1])
        bd = int(im[p[0]][p[1]][2]) - int(im[q[0]][q[1]][2])
        cdiff = math.sqrt(ld ** 2 + ad ** 2 + bd ** 2)
        return cdiff

    def gpq(p, q):
        return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2

    def weight(im, p, q, rc=7, rp=36): #rc가 작아지면 컬러 차이가 크면 차이 더 큼.
        if (p, q) in w_dp:
            return w_dp[p, q]
        w = math.exp(-(cpq(im, p, q) / rc + gpq(p, q) / rp)) * 255
        w_dp[(p, q)] = w
        # print(p,q,w)
        return w

    def dis_a(im1, p1, q1, im2, p2, q2, im1_g, im2_g):
        e0 = abs(int(im1_g[q1[0]][q1[1]]) - int(im2_g[q2[0]][q2[1]]))
        return weight(im1, p1, q1) * weight(im2, p2, q2) * e0

    def dis_b(im1, p1, q1, im2, p2, q2):
        return weight(im1, p1, q1) * weight(im2, p2, q2)


    def check_weight(test):
        ta = test[129:178, 334:383, ]  # 49 49
        tb = test[52:93, 121:162, ]  # 41 41
        tc = test[24:65, 302:343, ]  # 41 41
        td = test[160:201, 303:344]  # 41 41
        testers=[ta,tb,tc,td]
        index=1;
        for t in testers:
            height, width,c=t.shape
            center=(height//2,width//2)
            wgt=[]
            print(type(wgt),wgt)
            axs = plt.figure().subplots()
            axs.imshow(t)
            for i in range(height):
                for j in range(width):
                    wgt.append(round(weight(t,center,(i,j),rp=300,rc=80)))
                    print(wgt[-1],end=' ')
                print()
            wgt=np.array(wgt,dtype='uint8')
            wgt.shape=(height,width)
            print(wgt.shape,np.max(wgt),np.min(wgt),wgt.dtype)

            # ax = plt.figure().subplots()
            # ax.imshow(wgt)
            # #plt.imshow(wgt)
            # plt.show()
            cv2.imwrite("test{}.png".format(index), cv2.cvtColor(t,cv2.COLOR_LAB2BGR))
            cv2.imwrite("test_weight{}.png".format(index), wgt)
            cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
            cv2.namedWindow('original', cv2.WINDOW_NORMAL)
            cv2.imshow("original", cv2.cvtColor(t, cv2.COLOR_LAB2BGR))
            cv2.imshow("finalImg", wgt)
            cv2.waitKey(0)
            w_dp.clear()
            index+=1
    # check_weight(im)
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
            if len(w_dp)>700:
                w_dp.clear()
        print("{}th line finished".format(i))

    disparity = np.array(d,dtype='uint8')
    disparity.shape = (h - block_size+1, w - block_size+1)
    return disparity

def check_img_count(im):
    h,w=im.shape
    cnt=0
    for i in range(h):
        for j in range(w):
            print(im[i][j],end=' ')
            if im[i][j]<0:
                cnt+=1
        print()
    print('count: ',cnt)

def save_disparity(im,filename):
    M=np.max(im)
    if M!=1.0:
        im=im/M

    im = np.clip(im * 255, a_min=0, a_max=255).astype('uint8')
    cv2.imwrite("{}.png".format(filename), im)


def MSE(igt,im):
    h,w=im.shape
    mse= np.sum((igt.astype('int32')-im.astype('int32'))**2)/(h*w)
    print('mse: ',mse)
    return mse
def PSNR(im,mse):
    r=np.max(im)
    print(r,type(r))
    print('psnr',10*math.log((r**2)/mse,10))
    return 10*math.log((r**2)/mse,10)


if __name__ == '__main__':
    im = cv2.imread('tsukuba/scene1.row3.col1.ppm')  # left
    im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm')  # right
    igt = cv2.imread('tsukuba/truedisp.row3.col3.pgm',0)
    disl=matching_block(im,im2,block_size=5)
    disr=matching_block_rl(im2,im,block_size=5)
    print(disl.dtype,np.max(disl),np.min(disl))
    print(disr.dtype,np.max(disr),np.min(disr))
    cv2.imshow('l',disl)
    cv2.imshow('r',disr)
    cv2.imshow('out',disl-disr)
    #disparity=matching_for(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY), cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY))
    #disparity=matching_adaptive_weight(im2, im,block_size=33)
    #save_disparity(disparity,"test")
    #cv2.imshow("image1", im)
    #cv2.imshow("2", im2)
    #cv2.imshow("ds", disparity)
    #plt.imshow(igt,'igt')
    #plt.imshow(disparity,'gray')


    cv2.waitKey(0)
    plt.show()
