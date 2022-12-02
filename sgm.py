import numpy as np
import matplotlib.pyplot as plt
import cv2

im = cv2.imread('tsukuba/scene1.row3.col1.ppm',0)  # left
im2 = cv2.imread('tsukuba/scene1.row3.col2.ppm',0)  # right
igt = cv2.imread('tsukuba/truedisp.row3.col3.pgm',0)

P1=5
P2=150
maxd=15


def costvolume(maxd=15):
    global im,im2
    h, w = im.shape  # 288 384
    disparity = []
    for i in range(h):
        for j in range(w):
            d=[]
            for k in range(maxd):
                if j-k<0:
                    d.append(255)
                else:
                    d.append(abs(int(im[i][j]) - int(im2[i][j - k])))
            disparity+=d
    disparity = np.array(disparity)
    disparity.shape = (h, w,maxd)

    # cost volume을 통한 disparity map 확인
    # d=[]
    # for i in range(h):
    #     for j in range(w):
    #         d.append(np.argmin(disparity[i,j,:]))
    # d = np.array(d)
    # d.shape = (h, w)
    # cv2.imshow('cv',d/np.max(d))
    # cv2.waitKey(0)

    return disparity  # / np.max(disparity)


def get_direction_pairs(p,r):
    dir_range = []
    y, x = p
    while 0 < y < h - 1 and 0 < x < w - 1:
        dir_range.append(((y, x)))
        y += r[0]
        x += r[1]
    if y > h - 1 or y<0 or x< 0 or x > w - 1:
        y,x=dir_range.pop(-1)
    return (y,x),dir_range[::-1]


def get_lr(cv, p, r): #d별 lr
    global maxd
    prev_i,dir_path = get_direction_pairs(p, r)
    prev = cv[prev_i[0],prev_i[1],:]
    for e in dir_path:
        cur=cv[e[0],e[1],:].astype('int64')
        ub=min(prev)
        cur[0]+=min(prev[1]+P1,ub+P2)
        for i in range(1,maxd-1):
            cur[i]+=min(prev[i],prev[i-1]+P1,prev[i+1]+P1,ub+P2)
        cur[-1]+=min(prev[-2]+P1,ub+P2)
        prev=np.subtract(cur,ub)
    return prev

cv=costvolume(maxd)

def agg_cost_disp(p): #s를 통해 구한 d
    global cv
    directions=[(0,1),(1,0),(1,1),(-1,1),(1,2),(2,1),(-2,1),(-1,2)]
    s=np.zeros(maxd)
    for r in directions:
        s+=get_lr(cv, p, r)
        r = (-r[0], -r[1])
        s+=get_lr(cv, p, r)
    return np.argmin(s)

h,w=im.shape
disparity=[]
for i in range(h):
    for j in range(w):
        disparity.append(agg_cost_disp((i,j)))
        print(i,j)
disparity=np.array(disparity)
disparity.shape=(h,w)
cv2.imshow("dis",disparity/np.max(disparity))
cv2.waitKey(0)


