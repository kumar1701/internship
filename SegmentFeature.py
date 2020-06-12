import cv2
import numpy as np
from whichsegment import *
from All_features import *
from breakoutput import  *
import statistics

img=cv2.imread("i9.jpeg",1)     # reading input image
simg=cv2.imread("out.png",1)    # segmented image
output,n=WSegment(simg) 
output,n=breakoutput(output,n)         # each pixel belongs to which matrix and number of segment 
allfeature=All_features(img)    # calculate all features of image
segFeature=np.zeros((n-1,30,3)) #output of segment wise feature (no. of segment,no. of features,(mean,max,median))
for i in range(1,n):            # no. of segment
	temp=np.zeros((1,30))       # empty array to store the all features of pixel belongs to ith segment
	for j in range(0,200): 
		for k in range(0,200):
			if output[j][k]==i:
				t=allfeature[j*200+k,:]
				t=np.reshape(t,(1,30))
				temp=np.append(temp,t,0) # save all features of pixel in temp array
	#print(temp.shape)
	for l in range(0,30):  # for each feature we calculate
 		t=temp[1:,l]
 		#print(len(t))
		maxv=max(t)	  # max
		meanv=statistics.mean(t) # mean
		medianv=statistics.median(t) # median
		segFeature[i-1][l][0]=meanv	# store in i-1 th segment and l th feature and zero index
		segFeature[i-1][l][1]=maxv  # store in i-1 th segment and l th feature and first index
		segFeature[i-1][l][2]=medianv # store in i-1 th segment and l th feature and second dis
		
print(segFeature)	
		
