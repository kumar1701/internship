import numpy as np
import cv2
def findEdgeBoxes(img,dims):
	print('Finding Edge Boxes...')
	
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
	mask=np.zeros(dims)
	for i in range(0,len(boxes[0])):
		x,y,w,h=boxes[0][i]
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
		mask[y:y+h,x:x+w]=mask[y:y+h,x:x+w]+boxes[1][i]
	
	#cv2.imshow("output",img)
	mask=mask/np.amax(mask)	
	#cv2.imshow("mask",mask)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#print(mask)
	feature=mask.flatten()
	#print(feature)
	return(feature)
	
	
