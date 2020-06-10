import cv2
import numpy as np
import math
import numpy.matlib 
from findSaliency import *
from findEdgeBoxesTop20 import *
from findEdgeBoxes import *
from numpy import unravel_index
from operator import itemgetter

def matlab_style_gauss2D(shape,sigma):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
    
def  GetShiftedReverseGaussian(dims, peak_x, peak_y):

	new_dims = [dims[0]*2,dims[1]*2]

	SIGMA = 0.35 * math.sqrt(new_dims[1] * new_dims[0])
	out_filter = matlab_style_gauss2D(new_dims,SIGMA)
	out_filter = 1 - out_filter / np.amax(out_filter)

	# translate and crop
	x_center = round(new_dims[1]/2) + round(dims[1]/2) - peak_x
	y_center = round(new_dims[0]/2) + round(dims[0]/2) - peak_y
	x_start = int(x_center - round(dims[1]/2))
	y_start = int(y_center - round(dims[0]/2))
	x_end = int(x_start + dims[1] )
	y_end = int(y_start + dims[0] )
	out_filter = out_filter[y_start:y_end, x_start:x_end]
	return out_filter

def findAttenuatedSaliencyFromPeak(img,dims):
	
	print("find Attenuated Saliency From Peak.....")
	features=np.zeros((dims[0]*dims[1],3))
	features[:,0]=findSaliency(img,dims)
	features[:,1]=findEdgeBoxes(img,dims)
	features[:,2]=findEdgeBoxesTop20(img,dims)


	new_features = np.zeros((features.shape[0],features.shape[1]))
	
	for ii in range(0,features.shape[1]):
		current_feature = np.reshape(features[:, ii], dims)
		maxindex = current_feature.argmax()
		[peak_y ,peak_x] =unravel_index(current_feature.argmax(),dims)
		current_mask = GetShiftedReverseGaussian(dims, peak_x, peak_y)
		current_feature = current_feature * current_mask
		max_current_feature = np.amax(current_feature)
  		if max_current_feature==0:
			max_current_feature = 1
		current_feature = current_feature / max_current_feature
		new_features[:, ii] = current_feature.flatten()

	return new_features  











