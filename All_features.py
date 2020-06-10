import cv2
import numpy as np
from findText import *
from findSubBand import *
from findSaliency import *
from findHorizon import *
from findEdgeBoxesTop20 import *
from findEdgeBoxesNotInCentre import *
from findEdgeBoxes import *
from findDisToEdgeFeatures import *
from findDistToCenterFeatures import *
from findColorFeatures import *
from findAttenuatedSaliencyFromPeak import *
from findAttenuatedSaliency import *
from numpy import savetxt

img=cv2.imread("i1.jpeg",1)
#cv2.imshow("input",img)
dims=(200,200)

all_features=np.zeros((dims[0]*dims[1],30))

feature=findDistToEdgeFeatures(img,dims) # distance to edge

all_features[:,0]=feature

feature=findDistToCenterFeatures(img,dims) # distance to center

all_features[:,1]=feature
 
feature=findEdgeBoxes(img,dims) # find edge boxes

all_features[:,2]=feature

feature=findEdgeBoxesNotInCentre(img,dims) # find edge box not in center

all_features[:,3]=feature[:,0] # mask_sum
all_features[:,4]=feature[:,1] #mask_max

feature=findEdgeBoxesTop20(img,dims) # find Edge Box with top 20 score

all_features[:,5]=feature

feature=findSaliency(img,dims) # find saliency

all_features[:,6]=feature

feature=findAttenuatedSaliency(img,dims) #find Attenuated Saliency using center dis

all_features[:,7]=feature[:,0] #saliency
all_features[:,8]=feature[:,1] #edgebox
all_features[:,9]=feature[:,2] #edgeboxtop20

feature=findAttenuatedSaliencyFromPeak(img,dims) #find Attenuated Saliency using peak dis

all_features[:,10]=feature[:,0] #saliency
all_features[:,11]=feature[:,1] #edgebox
all_features[:,12]=feature[:,2] #edgeboxtop20

feature=findHorizon(img,dims) # find horizon

all_features[:,13]=feature

feature= findText(img,dims) # find text 

all_features[:,14]=feature

feature=findSubBands(img,dims) # find sub bands 
for i in range (0,feature.shape[1]):
	all_features[:,15+i]=feature[:,i] # at max 4 sub Bands


feature=findColor(img,dims)# find color features

for i in range (0,11):
	all_features[:,19+i]=feature[:,i]


savetxt('data.csv', all_features, delimiter=',')
#print(all_features)










