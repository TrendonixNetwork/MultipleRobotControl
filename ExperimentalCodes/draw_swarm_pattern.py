import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

drawing = False # true if mouse is pressed
ix,iy = -1,-1
#mode = True
# mouse callback function
pattern_coords=[]
h=400
w=500
robot_coords_initial=[]
max_robots=10
arena=np.ones((h,w,3),np.uint8)
h_r=np.random.randint(0,h-1,max_robots)
w_r=np.random.randint(0,w-1,max_robots)
for i in range(0,max_robots):
	
	print "h_r",h_r[i]
	print "w_r",w_r[i]
	robot_coords_initial.append((w_r[i],h_r[i]))
	print "robot_coords_initial",robot_coords_initial
	cv2.rectangle(arena,(w_r[i]-5,h_r[i]-5),(w_r[i]+5,h_r[i]+5),(0,255,0),3)
cv2.imshow('Arena',arena)
cv2.waitKey(0)

def draw_circle(event,x,y,flags,param):
	global ix,iy,drawing,mode
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix,iy = x,y
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			#img1=img2
			#if mode == True:
			#	cv2.rectangle(img1,(ix,iy),(x,y),(0,255,0),1)
			#else:
			cv2.circle(img,(x,y),5,(0,0,255),1)
			pattern_coords.append((x,y))
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		#if mode == True:
		#	cv2.rectangle(img1,(ix,iy),(x,y),(0,255,0),1)
		#else:
		cv2.circle(img,(x,y),5,(0,0,255),1)
		pattern_coords.append((x,y))

img = np.zeros((h,w,3), np.uint8)
img1=img.copy()
img2=img1.copy()
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
	cv2.imshow('image',img)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('m'):
		mode = not mode
	elif k == 27:
		break

# arena = np.zeros((h,w,3), np.uint8)
# for j in range(0,len(pattern_coords)):
# 	arena[pattern_coords[j][1],pattern_coords[j][0]]=255,255,255
# cv2.imshow('arena',arena)

# print "Total no. of points : ",len(pattern_coords)
# print len(pattern_coords)/max_robots

kmeans = KMeans(n_clusters=10)
kmeans.fit(pattern_coords)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["g.","r.","c.","y."]

for i in range(len(pattern_coords)):
	print("coordinate:",pattern_coords[i], "label:", labels[i])
	plt.plot(pattern_coords[i][0], pattern_coords[i][1], "g.", markersize = 10)
	#cv2.circle(img,(pattern_coords[i][0], pattern_coords[i][1]), 10, (255,0,0), -1)

for j in range(len(centroids)):
	cv2.circle(img,(int(centroids[j][0]), int(centroids[j][1])), 10, (255,0,0), -1)


cv2.imshow('clusttered plot',img)
#plt.scatter(centroids[:, 1],centroids[:, 0], marker = "x", s=150, linewidths = 5, zorder = 10)
#plt.scatter(centroids[:, 1],centroids[:, 0], marker = "x", s=150, linewidths = 5, zorder = 10)

C = cdist(robot_coords_initial, centroids)
_, assignment = linear_sum_assignment(C)

plt.axis([0,w,h,0])

for p in range(max_robots):
	plt.plot(robot_coords_initial[p][0], robot_coords_initial[p][1],'bo', markersize = 10)
	plt.plot(centroids[p][0], centroids[p][1],'rs',  markersize = 7)
	plt.plot((robot_coords_initial[p][0], centroids[assignment[p]][0]), (robot_coords_initial[p][1], centroids[assignment[p]][1]), 'k')
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.axes().set_aspect('equal')

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()