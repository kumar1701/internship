import cv2
import sys
import numpy as np

def intersection(lst1, lst2):
	lst3=np.zeros((len(lst1)), dtype=bool) 
	for i in range(0,len(lst1)):
		if lst1[i]==True and lst2[i]==True:
			lst3[i]=True
		else :
			lst3[i]=False	
    #lst3 = [value for value in lst1 if value in lst2] 
	return lst3 


def colhist(img,m):
	
	if m>1:
		img[:,:,0] = cv2.medianBlur(img[:,:,0],m)
		img[:,:,1] = cv2.medianBlur(img[:,:,1],m)
		img[:,:,2] = cv2.medianBlur(img[:,:,2],m)
	
	img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	r=img[:,:,0]
	g=img[:,:,1]
	b=img[:,:,2]
	r=r.flatten()
	g=g.flatten()	
	b=b.flatten()	
	
	bins=10
	H=np.zeros((bins,bins,bins))
	
	bucketsize=0.1
	for i in range (0,len(r)):
		if r[i]<=0:
			r[i]=0
		elif r[i]>0.999:
			r[i]=0.999
	for i in range (0,len(g)):
		if g[i]<=0:
			g[i]=0
		elif g[i]>0.999:
			g[i]=0.999
	for i in range (0,len(b)):
		if b[i]<=0:
			b[i]=0
		elif b[i]>0.999:
			b[i]=0.999

	for ri in range(0,bins):
		for gi in range(0,bins):
			for bi in range(0,bins):
				rmin=ri*bucketsize
	        	rmax=(ri+1)*bucketsize
	        	gmin=(gi)*bucketsize
	        	gmax=(gi+1)*bucketsize
	        	bmin=(bi)*bucketsize
	        	bmax=(bi+1)*bucketsize
	       		rbool=np.logical_and(r>=rmin,r<rmax)
	        	gbool=np.logical_and(g>=gmin,g<gmax)
	       		bbool=np.logical_and(b>=bmin,b<bmax)
	        	I=intersection(intersection(rbool, gbool), bbool)
	        	rInBucket=r[I]
	        	H[ri, gi, bi]=len(rInBucket)
	
	H=H+1;
	#print(H)
	Hpercent=H/(sum(sum(sum(H))))
	#print(Hpercent)
	Hprob=-np.log(Hpercent)
	#print(Hprob)
	feature=np.zeros((len(r)))
	for i in range (0,len(r)):
		ri=int(round(r[i]/bucketsize-1))
		gi=int(round(g[i]/bucketsize-1))
		bi=int(round(b[i]/bucketsize-1))
		feature[i]=Hprob[ri, gi, bi]
	
	#print(feature)
	return feature



