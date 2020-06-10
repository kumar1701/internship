import cv2
import pytesseract
from pytesseract import Output
import numpy as np

def findText(img,dims):
	
	print("finding Text...")
	img=cv2.resize(img,dims)
	data=pytesseract.image_to_boxes(img)
	mask=np.zeros((img.shape[0],img.shape[1]))
	d = pytesseract.image_to_data(img, output_type=Output.DICT)
	n_boxes = len(d['level'])
	#print(n_boxes)
	for i in range (0,n_boxes):
		(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		mask[y:y+h,x:x+w]=1;
		#cv2.imshow("output",img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()
			
	#cv2.imshow("output",img)
	#cv2.imshow("mask",mask)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	feature=mask.flatten()
	return(feature)	
		
		
