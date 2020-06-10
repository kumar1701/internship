import cv2
import numpy as np
import pyPyrTools as ppt
import scipy


def fspecial_gauss(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()
    
def findSubBands(img,dims):

	print("find subband...")
	img=cv2.resize(img,dims)
	height=4
	edges='reflect1'
	filter='sp3Filters'
	pyr = ppt.Spyr(cv2.blur(img,(3,3)), height, 'sp3Filters',edges)
	
	feature=np.zeros((dims[0]*dims[1],pyr.spyrHt()))
	for s in range(pyr.spyrHt()):
		band = pyr.spyrBand(s,0)
		band=abs(band)
		f_special = fspecial_gauss(6,2)
		scipy.ndimage.convolve(band,f_special, mode='nearest')
		x=np.mean(band)
		band=band/x
		band=cv2.resize(band,dims)
		feature[:,s]=band.flatten()
		
	#print(feature)
	return(feature)	
