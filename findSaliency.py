import cv2
import numpy as np

def findSaliency(img,dims):
	
	print("finding saliency.......")	
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(img)
	
	saliencyMap=cv2.resize(saliencyMap,dims)
	
	feature=saliencyMap.flatten()
	#print(feature)
	return feature

