from __future__ import print_function

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import interpolation,filters

def scale_to_h(img,target_height,order=1,dtype=np.dtype('f'),cval=0):
    h,w = img.shape
    scale = target_height*1.0/h
    target_width = int(scale*w)
    output = interpolation.affine_transform(1.0*img,np.eye(2)/scale,order=order,
                                            output_shape=(target_height,target_width),
                                            mode='constant',cval=cval)
    output = np.array(output,dtype=dtype)
    return output

class CenterNormalizer(object):
    def __init__(self,target_height=48,params=(4,1.0,0.3)):
        self.debug = int(os.getenv("debug_center") or "0")
        self.target_height = target_height
        self.range,self.smoothness,self.extra = params
        print("# CenterNormalizer")
    def setHeight(self,target_height):
        self.target_height = target_height
    def measure(self,line):
        h,w = line.shape
        smoothed = filters.gaussian_filter(line,(h*0.5,h*self.smoothness),mode='constant')
        smoothed += 0.001*filters.uniform_filter(smoothed,(h*0.5,w),mode='constant')
        self.shape = (h,w)
        a = np.argmax(smoothed,axis=0)
        a = filters.gaussian_filter(a,h*self.extra)
        self.center = np.array(a,'i')
        deltas = np.abs(np.arange(h)[:,np.newaxis]-self.center[np.newaxis,:])
        self.mad = np.mean(deltas[line!=0])
        self.r = int(1+self.range*self.mad)
        if self.debug:
            plt.figure("center")
            plt.imshow(line,cmap=plt.cm.gray)
            plt.plot(self.center)
            plt.ginput(1,1000)
    def dewarp(self,img,cval=0,dtype=np.dtype('f')):
        assert img.shape==self.shape
        h,w = img.shape
        # The actual image img is embedded into a larger image by
        # adding vertical space on top and at the bottom (padding)
        hpadding = self.r # this is large enough
        padded = np.vstack([cval*np.ones((hpadding,w)),img,cval*np.ones((hpadding,w))])
        center = self.center + hpadding
        dewarped = [padded[center[i]-self.r:center[i]+self.r,i] for i in range(w)]
        dewarped = np.array(dewarped,dtype=dtype).T
        return dewarped
    def normalize(self,img,order=1,dtype=np.dtype('f'),cval=0):
        dewarped = self.dewarp(img,cval=cval,dtype=dtype)
        h,w = dewarped.shape
        # output = zeros(dewarped.shape,dtype)
        scaled = scale_to_h(dewarped,self.target_height,order=order,dtype=dtype,cval=cval)
        return scaled


