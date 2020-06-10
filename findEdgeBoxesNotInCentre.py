import numpy as np
import cv2
from operator import itemgetter

def findEdgeBoxesNotInCentre(img,dims):
	print('Finding Edge Boxes not in centre...')
	
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
	edge_boxes.setMinScore(0.015)
	boxes = edge_boxes.getBoundingBoxes(edges, orimap)
	
	points=boxes[0]
	score=boxes[1]
	mat=np.hstack((points,score))
	mat=sorted(mat, key =itemgetter(4),reverse=True)
	

	mask_sum=np.zeros(dims)
	mask_max=np.zeros(dims)
	for i in range(0,len(mat)):
		x1=int(mat[i][0])
		y1=int(mat[i][1])
		w=int(mat[i][2])
		h=int(mat[i][3])
		score=mat[i][4]
		x2=x1+w
		y2=y1+h
		
		ALLOWED_AREA = 0.25
		if not((x1<ALLOWED_AREA*dims[1] and x2<ALLOWED_AREA*dims[1]) or(x1>(1-ALLOWED_AREA)*dims[1] and x2>(1-ALLOWED_AREA)*dims[1]) or (y1<ALLOWED_AREA*dims[0] and y2<ALLOWED_AREA*dims[0]) or (y1>(1-ALLOWED_AREA)*dims[0] and y2<(1-ALLOWED_AREA)*dims[0])):
			continue
		
		SIZE_THRESHOLD = 0.05 # 5 percent
		if w*h>SIZE_THRESHOLD*dims[0]*dims[1]:
			continue
		
		# Only keep rectangles that touch BOUNDARY_WIDTH (measured in pixels)
		BOUNDARY_WIDTH = 5
		if not(x1<=BOUNDARY_WIDTH or y1<=BOUNDARY_WIDTH or x2>dims[1]-BOUNDARY_WIDTH or y2>dims[0]-BOUNDARY_WIDTH):
			continue
		
		mask_sum[y1:y2,x1:x2]=mask_sum[y1:y2,x1:x2]+score
		for j in range(y1,y2):
			for k in range(x1,x2):
				mask_max[j,k]=max(mask_max[j,k],score)
		
		cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
	
	if np.amax(mask_sum)!=0:	
		mask_sum=mask_sum/np.amax(mask_sum)
	if np.amax(mask_max)!=0:	
		mask_max=mask_max/np.amax(mask_max)		
	
	#cv2.imshow("output",img)
	#cv2.imshow("mask",mask_sum)
	#cv2.imshow("mask_max",mask_max)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#print(mask_sum[:,6])
	#print(mask_max[:,6])
	feature=np.zeros((dims[0]*dims[1],2))
	feature[:,0]=mask_sum.flatten()
	feature[:,1]=mask_max.flatten()
	#print(feature)
	return(feature)
