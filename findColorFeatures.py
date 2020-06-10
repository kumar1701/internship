import cv2
import numpy as np
import matplotlib.pyplot as plt
from colhist import *
from scipy.interpolate import interp1d

def findColor(img,dims):
	
	print("find color ....")
	img=cv2.resize(img,(dims))
	
	#cv2.imshow("input",img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	features=np.zeros((dims[0]*dims[1],11))
	x=[]
	gap=255/float(101)
	for i in range (0,99):
		x.append(i*gap)
	x.append(255)
	#print(len(x))
	channel=img[:,:,0]
	hist = cv2.calcHist([img],[0], None, [100], [0, 256])
	hist=hist+1
	hist=hist/sum(hist)
	prob=-np.log(hist)
	prob=np.asarray(prob).squeeze()
	set_interp = interp1d(x, prob, kind='linear')
	prob = set_interp(channel.flatten())
	prob=(prob-min(prob))/max(prob)
	#print(prob)
	features[:,3]=prob
	 
	channel=img[:,:,1]
	hist = cv2.calcHist([img], [1], None, [100], [0, 256])
	hist=hist+1
	hist=hist/sum(hist)
	prob=-np.log(hist)
	prob=np.asarray(prob).squeeze()
	set_interp = interp1d(x, prob, kind='linear')
	prob = set_interp(channel.flatten())
	prob=(prob-min(prob))/max(prob)
	#print(prob)
	features[:,4]=prob
	
	
	channel=img[:,:,2]
	hist = cv2.calcHist([img], [2], None, [100], [0, 256])
	hist=hist+1
	hist=hist/sum(hist)
	prob=-np.log(hist)
	prob=np.asarray(prob).squeeze()
	set_interp = interp1d(x, prob, kind='linear')
	prob = set_interp(channel.flatten())
	prob=(prob-min(prob))/max(prob)
	features[:,5]=prob
	#print(prob)
	
	mF=[1,3,5,9,17]	
	for i in range(0,5):
		f=colhist(img,mF[i])
		#print(f)
		features[:,i+6]=f
	
	img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
	channel=img[:,:,0]
	features[:,0]=channel.flatten()
	channel=img[:,:,1]
	features[:,1]=channel.flatten()
	channel=img[:,:,2]
	features[:,2]=channel.flatten()	
	
	
	return(features)			

