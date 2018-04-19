import tensorflow as tf
import numpy as np

import os
import sys
import time
import random
import math
import numpy as np
#import skimage.io
import imageio
import cv2
import tqdm
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.io as sio

import visualize

def main():
    
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    # video directory
    video_dir = "../../../../aic2018/track1/track1_videos/"
    detect_dir = "../../../../aic2018/track1/detect/"
    save_dir = "../../../../aic2018/track1/detect_videos/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # read transform matrix
    M = sio.loadmat('track1_M.mat')
    keys = list(M.keys())
    for k in keys:
        if not k.startswith('M'):
            M.pop(k, None)
    # match videos and M matrices
    videos = sorted([(i, x) for i, x in enumerate(sorted(os.listdir(video_dir))) if x.endswith('.mp4')])
    Ms = dict()
    for k in M.keys():
        m = M[k]
        t = k.replace('M', '').split('_')
        if len(t) == 1:
            s, e = t[0], t[0]
        else:
            s, e = t
        s, e = int(s), int(e)
        for i in range(s, e+1):
            Ms[videos[i][1]] = m
            
    def savefig(x):
        fnum = x
        videoname = [x for x in os.listdir(video_dir) if x.startswith("Loc2_3")][0]
        print(videoname)
        print("Processing video {}...".format(videoname))
        # read video
        video_file = os.path.join(video_dir, videoname)
        vid = imageio.get_reader(video_file,  'ffmpeg')
        # load pkl files
        pkl_dir = os.path.join(detect_dir, videoname)
        pkl_file = str(x).zfill(7)+".pkl"#sorted([x for x in os.listdir(pkl_dir)])
        r = pickle.load(open(os.path.join(pkl_dir,pkl_file), "rb"))
        bottom_contour = [visualize.extract_bottom(c) for i, c in enumerate(r['contours']) 
                          if class_names[r['class_ids'][i]] in ['car', 'motorcycle', 'bus', 'train', 'truck']]

        # visualize mask_image
        # 2d to 3d
        def coord2dto3d(m, pts):
            """
            m: [3,3]
            pts: [n, 3]
            """
            X = pts # (u,v,1)
            N = 2
            U1 = np.dot(X, m) # transform in homogeneous coordinates
            UN = np.tile(U1[:, -1][:,np.newaxis], (1, N)) # replicate the last column of U
            U = U1[:, :-1] / UN
            return U

        m = Ms[videoname]
        pt3d = []
        colors = []
        color_list = ['r', 'g', 'b']
        for i, objc in enumerate(bottom_contour):
            length = 0
            for c in objc:
                if c.shape[0] > 1:
                    c = np.squeeze(c)
                else:
                    c = c[0]
                c = np.concatenate((c, np.ones((c.shape[0], 1))), axis=1)
                c3d = coord2dto3d(m, c)
                pt3d.append(c3d)
                length += c.shape[0]
            colors += [color_list[i%3]]*length 

        pt3d_np = np.concatenate(pt3d, axis=0)
        assert pt3d_np.shape[0] == len(colors), (pt3d_np.shape[0], len(colors))

        fig, ax = plt.subplots()
        ax.set_xlim([-50, 100])
        ax.set_ylim([-20, 180])
        ax.scatter(pt3d_np[:, 0], pt3d_np[:, 1], c=colors)
        #plt.axis('equal')
        plt.savefig("./save/{}.png".format(str(x).zfill(7)))
        print("./save/{}.png".format(str(x).zfill(7)))
        plt.close(fig) 

    for i in range(100):
        savefig(i)
    
if __name__ == '__main__':
    main()
    
