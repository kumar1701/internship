import cv2
import numpy as np

def breakoutput(output,n):
	x=n
	dic=[[1,0,-1,0,-1,-1,1,1],[0,1,0,-1,-1,1,1,-1]]
	n=n-1
	for i in range(1,x):
		my_map=dict()
		count=0
		for j in range(0,200):
			for k in range(0,200):
				x=tuple((j,k))
				if output[j][k]==i and x not in my_map:
					count=count+1
					if count>1:
						n=n+1
					#print(x,n,i)
					queue=[]
					queue.append(x)
					if count>1:
						output[j][k]=n+1
						
					my_map[x]=1
					while len(queue)>0:
						y,z=queue.pop(0)
						for l in range(0,8):
							nx=y+dic[0][l]
							ny=z+dic[1][l]
					 		nt=tuple((nx,ny))
					 		if nt not in my_map and nx>=0 and ny>=0 and nx<200 and ny<200 and output[nx][ny]==i:
					 			queue.append(nt)
					 			my_map[nt]=1
					 			if count>1 : 
					 				output[nx][ny]=n
	
	return output,n+1		 		
	
