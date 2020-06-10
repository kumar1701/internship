import numpy as np
import cv2
from operator import itemgetter

def findEdgeBoxesTop20(img,dims):
	print('Finding Edge Boxes Top 20...')
	
	img=cv2.resize(img,dims)
	model=('model.yml') 
	edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
	rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0) # normalization(0-1)

	orimap = edge_detection.computeOrientation(edges)
	edges = edge_detection.edgesNms(edges, orimap)
	
	edge_boxes = cv2.ximgproc.createEdgeBoxes()
	edge_boxes.setMaxBoxes(10000)
	edge_boxes.setAlpha(0.65)
	edge_boxes.setBeta(0.75)
	edge_boxes.setMinScore(0.01)
	boxes = edge_boxes.getBoundingBoxes(edges, orimap)
	
	points=(boxes[0])
	score1=boxes[1]
	mat=np.hstack((points,score1))
	mat=sorted(mat,key=itemgetter(4),reverse=True)#sending order according to score
	
	k=20#top 20 boxes
	mask=np.zeros(dims)
	
	for i in range(0,20):
		x=int(mat[i][0])
		y=int(mat[i][1])
		w=int(mat[i][2])
		h=int(mat[i][3])
		score=mat[i][4]
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
		mask[y:y+h,x:x+w]=mask[y:y+h,x:x+w]+score
  	
	mask=mask/np.amax(mask)
	#cv2.imshow("img",img)
	#cv2.imshow("mask",mask)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#print(mask)
	feature=mask.flatten()
	#print(feature)
	return(feature)
