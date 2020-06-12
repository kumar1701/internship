import numpy as np
import cv2 
import math
from skimage.measure import label,regionprops

def segmentspecificfeature(img):
	print("calculating the segment specific feature.....")
	region_pros=np.zeros((12))
	gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
	#cv2.imshow("mask",gray_img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	label_gray_img = label(gray_img, connectivity=gray_img.ndim)
	print(label_gray_img)
	print(gray_img.shape)
	#label_img=	label(img, connectivity=img.ndim)
	props = regionprops(label_gray_img)[0]  # only one object
	#props1 = regionprops(label_img)[0]
	
	euler=props.euler_number
	#print(euler)
	region_pros[0]=euler
	filledArea=props.filled_area
	#print(filledArea)
	region_pros[1]=filledArea
	convexarea=props.convex_area #convex area
	#print(convexarea)
	region_pros[2]=convexarea
	area=props.area # area
	#print(area)
	region_pros[3]=area
	eccentricity=props.eccentricity# eccentricity
	#print(eccentricity)
	region_pros[4]=eccentricity
	major_axis_length=props.major_axis_length # major_axis_length
	#print(major_axis_length)
	region_pros[5]=major_axis_length
	minor_axis_length=props.minor_axis_length # minor_axis_length
	#print(minor_axis_length)
	region_pros[6]=minor_axis_length
	perimeter=props.perimeter # perimeter
	#print(perimeter)
	region_pros[7]=perimeter
	solidity=props.solidity # solidity
	#print(solidity)
	region_pros[8]=solidity
	orientation=props.orientation # orientation
	#print(orientation)
	region_pros[9]=orientation
	equivalent_diameter=props.equivalent_diameter # equivalent diameter
	#print(equivalent_diameter)
	region_pros[10]=equivalent_diameter
	extent=props.extent # extent
	#print(extent)
	region_pros[11]=extent
	return region_pros
