import numpy as np
import cv2


def findDistToEdgeFeatures(img,dims):
	print('Finding distance to the Edge...')
	# calculate the feature.  Each pixel is the distance away from the
	# center (should measure from the original image)
	imgr = img.shape[0]
	imgc = img.shape[1]
	distMatrix= np.zeros((imgr, imgc))
	for x in range(0,imgr):
		for y in range(0,imgc):
			distMatrix[x,y] = min(min(x,y),min(imgr-1-x,imgc-1-y))
	
	distMatrix = distMatrix/np.amax(distMatrix)
	distMatrix = cv2.resize(distMatrix,dims,interpolation=cv2.INTER_LINEAR)
	#cv2.imshow("edge_dist.png",distMatrix)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	features=distMatrix.flatten()
	#print(features)
	return(features)

	

