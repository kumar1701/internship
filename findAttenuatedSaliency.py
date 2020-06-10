import cv2
import numpy as np
import math
import numpy.matlib 
from findSaliency import *
from findEdgeBoxesTop20 import *
from findEdgeBoxes import *
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

def findAttenuatedSaliency(img,dims):

	print("find attenuated saliency with distance from center...")
	features=np.zeros((dims[0]*dims[1],3))
	features[:,0]=findSaliency(img,dims)
	features[:,1]=findEdgeBoxes(img,dims)
	features[:,2]=findEdgeBoxesTop20(img,dims)

	SIGMA = 0.7 * math.sqrt(dims[0]*dims[1])
	attenuating_feature_centered = matlab_style_gauss2D(dims, SIGMA)

	attenuating_feature_centered = 1 - attenuating_feature_centered/ np.amax(attenuating_feature_centered)

	attenuating_feature_centered = np.reshape(attenuating_feature_centered,(dims[0]*dims[1],1))

	features = features * np.matlib.repmat(attenuating_feature_centered,1,3)
	max_of_features = np.amax(features)
	if max_of_features==0:
		max_of_features=1

	features = features /max_of_features
	#print(features)
	return(features)	
