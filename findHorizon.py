import cv2
import numpy as np

def findHorizon(img,dims):
	
	print("find horizon")
	img=cv2.resize(img,dims)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	
	#minLineLength = 100
	Gap = 50
	lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180,threshold=40,maxLineGap=Gap)
	#print(len(lines[0]))
	for x1,y1,x2,y2 in lines[0]:
		#print(x1,y1,x2,y2)
		cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
		
	mask=np.zeros((dims[0],dims[1]))
	x=[0,255,0]
	x=tuple(x)
	for i in range(0,200):
		for j in range(0,200):
			if tuple(img[i][j])==x:
				mask[i][j]=1
			
	feature=mask.flatten()			
	return(feature)
	

