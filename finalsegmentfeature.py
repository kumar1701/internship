import cv2
import numpy as np
from whichsegment import *
from All_features import *
from breakoutput import  *
from regionpros import *
from distactorsegment import *
import statistics
import glob
import time

start_time=time.time()
#filenames = [img for img in glob.glob("intersection_annotations/0.jpeg")]
#q=1
#pro=1
#for n in filenames:
	#if pro<=421:
	#	pro=pro+1
		#if(pro==44):
	#	q=q+1
	#	continue
	#anoted= cv2.imread(n,1)
	#iname="input_datasets/"+n[25:]
	#sname="segmented/"+n[25:len(n)-5]+"_segmented.jpeg"
	#print(iname,sname)
	#mean_name="mean_intersection/"+n[25:len(n)-5]+".csv"
	#max_name="max_intersection/"+n[25:len(n)-5]+".csv"
	#median_name="median_intersection/"+n[25:len(n)-5]+".csv"
img=cv2.imread("1239.jpeg",1)     # reading input image
simg=cv2.imread("1239_segmented.jpeg",1)   # segmented image
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#anoted=cv2.imread("1239.png",1)
#print(anoted.shape)
output,n=WSegment(simg) 
output,n,umap=breakoutput(output,n)         # each pixel belongs to which matrix and number of segment 
print(n)

allfeature=All_features(img)    # calculate all features of image
segFeature=np.empty((n,43,3),dtype=object) #output of segment wise feature (no. of segment,no. of features,(mean,max,median))
segFeature[0,0,:]="DisToEdge"
segFeature[0,1,:]="DisToCenter"
segFeature[0,2,:]="Edgebox"
segFeature[0,3,:]="EdgeboxNotInCentreMask"
segFeature[0,4,:]="EdgeboxNotInCentreMax"
segFeature[0,5,:]="EdgeboxTop20"
segFeature[0,6,:]="Saliency"
segFeature[0,7,:]="AttenuatedSaliencyUsingDistance"
segFeature[0,8,:]="AttenuatedEdgeboxUsingDistance"
segFeature[0,9,:]="AttenuatedEdgeboxTop20UsingDistance"
segFeature[0,10,:]="AttenuatedSaliencyUsingPeak"
segFeature[0,11,:]="AttenuatedEdgeboxUsingPeak"
segFeature[0,12,:]="AttenuatedEdgeboxTop20UsingPeak"
segFeature[0,13,:]="Horizon"
segFeature[0,14,:]="TextBox"
segFeature[0,15,:]="subBand0"
segFeature[0,16,:]="subBand1"
segFeature[0,17,:]="subBand2"
segFeature[0,18,:]="subBand3"
segFeature[0,19,:]="colorRed"
segFeature[0,20,:]="colorGreen"
segFeature[0,21,:]="colorBlue"
segFeature[0,22,:]="colorProbRed"
segFeature[0,23,:]="colorProbGreen"
segFeature[0,24,:]="colorProbBlue"
segFeature[0,25,:]="MediancolorProb1"
segFeature[0,26,:]="MediancolorProb2"
segFeature[0,27,:]="MediancolorProb3"
segFeature[0,28,:]="MediancolorProb4"
segFeature[0,29,:]="MediancolorProb5"
segFeature[0,30,:]="eulerNumber"
segFeature[0,31,:]="filledArea"
segFeature[0,32,:]="ConvexArea"
segFeature[0,33,:]="Area"
segFeature[0,34,:]="eccentricity"
segFeature[0,35,:]="major_axis_length"
segFeature[0,36,:]="minor_axis_length"
segFeature[0,37,:]="perimeter"
segFeature[0,38,:]="solidity"
segFeature[0,39,:]="orientation"
segFeature[0,40,:]="equi.Diameter"
segFeature[0,41,:]="extent"
segFeature[0,42,:]="Distractor"
	#segFeature[0,43,:]="DisToEdge"
print("find segment wise feature")
s_time=time.time()
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
		maxv=max(t) # max
		meanv=statistics.mean(t) # mean
		medianv=statistics.median(t) # median
		segFeature[i][l][0]=meanv	# store in i-1 th segment and l th feature and zero index
		segFeature[i][l][1]=maxv  # store in i-1 th segment and l th feature and first index
		segFeature[i][l][2]=medianv # store in i-1 th segment and l th feature and second dis
print(time.time()-s_time)		
print("find segment specific feature")
s_time=time.time()
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
	#print(masks.shape)
	#cv2.imshow("mask",masks)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	temp=segmentspecificfeature(masks)
	for l in range(0,12):
		segFeature[i][30+l][0]=temp[l]
		segFeature[i][30+l][1]=temp[l]
		segFeature[i][30+l][2]=temp[l]
print(time.time()-s_time)	

	#anoted_img=cv2.imread("",1)

#distractor=findDistractor(anoted,simg,output,n)	
#segFeature[1:,42,0]=distractor #mean=0
#segFeature[1:,42,1]=distractor # max=1
#segFeature[1:,42,2]=distractor # median=2	
#np.savetxt('x.csv', segFeature[:,:,0], fmt='%s')
#np.savetxt('y.csv', segFeature[:,:,1], fmt='%s')
#np.savetxt('z.csv', segFeature[:,:,2], fmt='%s')	
#savetxt("1239_mean_intersection.csv", segFeature[:,:,0],delimiter=',',fmt='%s')
#savetxt("1239_max_intersection.csv", segFeature[:,:,1], delimiter=',',fmt='%s')
#savetxt("1239_median_intersection.csv", segFeature[:,:,2], delimiter=',',fmt='%s')	
#print(time.time()-start_time)
img=cv2.resize(img,(200,200))
simg=cv2.resize(simg,(200,200))
new_img=img.copy()
dis=1
for  i in range(0,len(umap[dis])):
	x,y=umap[dis][i]
	new_img[x][y][0]=0
	new_img[x][y][1]=0
	new_img[x][y][2]=0

dis=5
for  i in range(0,len(umap[dis])):
	x,y=umap[dis][i]
	new_img[x][y][0]=0
	new_img[x][y][1]=0
	new_img[x][y][2]=0
cv2.imwrite("output.jpeg",new_img)

img=cv2.resize(img,(400,400))
simg=cv2.resize(simg,(400,400))
new_img=cv2.resize(new_img,(400,400))
cv2.imshow("img",img)
cv2.imshow("anoted_img",new_img)
cv2.imshow("simg",simg)
cv2.waitKey(0)
cv2.destroyAllWindows()
#print(segFeature)
	#print(q)
	#q=q+1
