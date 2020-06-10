import numpy as np
import cv2
import math


def findDistToCenterFeatures(img,dims):
	print('Finding distance to the center...')
	# calculate the feature.  Each pixel is the distance away from the
	# center (should measure from the original image)
	
	imgr = img.shape[0]
	imgc = img.shape[1]
	midpointx = round(imgr/2)
	midpointy = round(imgc/2)
	distMatrix= np.zeros((imgr, imgc))
	for x in range(0,imgr):
		for y in range(0,imgc):
			distMatrix[x,y] = round(math.sqrt((x-midpointx)**2 + (y-midpointy)**2))
	
	distMatrix = distMatrix/np.amax(distMatrix)
	distMatrix = cv2.resize(distMatrix,dims,interpolation=cv2.INTER_LINEAR)
	#cv2.imshow("center_dist.png",distMatrix)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	features = distMatrix.flatten()
	#print(features)
	return(features)

		
