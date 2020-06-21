import cv2
import numpy as np
from breakoutput import *
from whichsegment import *
import time
def findDistractor(annoted_img,segmented_img,output,n):
	print("find_distractor...")
	start_time=time.time()
	annoted_img=cv2.resize(annoted_img,(200,200))
	segmented_img=cv2.resize(segmented_img,(200,200))
	total_pixel=0
	anoted_pixel=0
	distractor=np.zeros((n-1))

	for i in range(1,n):
		total_pixel=0
		anoted_pixel=0
		z=1
		for j in range(0,200):
			#print(j)
			for k in range(0,200):
				#print(annoted_img[j][k])
				if output[j][k]==i:
			 		total_pixel=total_pixel+1
			 		if annoted_img[j][k][2]!=0: # and annoted_img[j][k][0]==0 and annoted_img[j][k][0]==0:
			 			anoted_pixel=anoted_pixel+1
			 		
		#print(i,anoted_pixel,total_pixel)		 		
		x=float(anoted_pixel)/float(total_pixel)
		if x>0.2:
			distractor[i-1]=1
		else:
			distractor[i-1]=0
	#print(time.time()-start_time)					
	return distractor			
