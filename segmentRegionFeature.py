import cv2
import numpy as np
from whichsegment import *
from breakoutput import  *
from newfile import *


img=cv2.imread("i10.jpeg",1)
img=cv2.resize(img,(200,200))
cv2.imshow("input",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
simg=cv2.imread("out10.png",1)

output,n=WSegment(simg)
output,n=breakoutput(output,n)
print(n)

segFeatures=np.zeros((n-1,12))
for i in range(1,n):
	masks=np.zeros((200,200,3),np.uint8)
	count=0
	x1=199
	y1=199
	x2=0
	y2=0
	for j in range(0,200):
		for k in range(0,200):
			if output[j][k]==i:
				count=count+1
				masks[j][k][0]=img[j][k][0]
				masks[j][k][1]=img[j][k][1]
				masks[j][k][2]=img[j][k][2]		
				x1=min(x1,j)
				y1=min(y1,k)
				x2=max(x2,j)
				y2=max(y2,k)			
	masks=masks[x1:x2+1,y1:y2+1]
	#print(count)
	#cv2.imshow("mask",masks)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#if count>10:
	temp=segmentspecificfeature(masks)
	segFeatures[i-1,:]=temp		
print(segFeatures)	
