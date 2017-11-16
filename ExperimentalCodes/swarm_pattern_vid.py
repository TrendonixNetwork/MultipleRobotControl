import cv2
import cv2.cv as cv
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
#import serial

drawing = False # true if mouse is pressed
ix,iy = -1,-1
#mode = True
# mouse callback function
pattern_coords=[]
robot_coords_initial=[]
max_robots=0

def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            #img1=img2
            #if mode == True:
            #   cv2.rectangle(img1,(ix,iy),(x,y),(0,255,0),1)
            #else:
            cv2.circle(draw_pad,(x,y),5,(0,0,255),1)
            pattern_coords.append((x,y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        #if mode == True:
        #   cv2.rectangle(img1,(ix,iy),(x,y),(0,255,0),1)
        #else:
        cv2.circle(draw_pad,(x,y),5,(0,0,255),1)
        pattern_coords.append((x,y))

kernel = np.ones((5,5),np.uint8)

# Take input from webcam
cap = cv2.VideoCapture(0)
time.sleep(2)
#ser = serial.Serial('COM4', 9600, timeout=0)
_,img_0=cap.read()
#print img_0.shape
h,w,c=img_0.shape

draw_pad = np.zeros((h,w,c), np.uint8)
draw_pad1=draw_pad.copy()
draw_pad2=draw_pad1.copy()
cv2.namedWindow('Draw any pattern you want to!')
cv2.setMouseCallback('Draw any pattern you want to!',draw_circle)

while(1):
    cv2.imshow('Draw any pattern you want to!',draw_pad)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
centre_x=w/2
centre_y=h/2
print w,h

c_centre=[]

cx1=0
cy1=0
# Reduce the size of video to 320x240 so rpi can process faster
#cap.set(3,320)
#cap.set(4,240)

def nothing(x):
    pass
# Creating a windows for later use
#cv2.namedWindow('HueComp')
#cv2.namedWindow('SatComp')
#cv2.namedWindow('ValComp')
#cv2.namedWindow('closing')
#cv2.namedWindow('tracking')
#cv2.namedWindow('frame')


# Creating track bar for min and max for hue, saturation and value
# You can adjust the defaults as you like
#cv2.createTrackbar('hmin', 'HueComp',12,179,nothing)
#cv2.createTrackbar('hmax', 'HueComp',37,179,nothing)

#cv2.createTrackbar('smin', 'SatComp',96,255,nothing)
#cv2.createTrackbar('smax', 'SatComp',255,255,nothing)

#cv2.createTrackbar('vmin', 'ValComp',186,255,nothing)
#cv2.createTrackbar('vmax', 'ValComp',255,255,nothing)

# My experimental values
# hmn = 12
# hmx = 37
# smn = 145
# smx = 255
# vmn = 186
# vmx = 255
cv2.namedWindow('Color 1')
cv2.createTrackbar('H1 Lower','Color 1',22,180,nothing) #27
cv2.createTrackbar('H1 Higher','Color 1',60,180,nothing) #58
cv2.createTrackbar('S1 Lower','Color 1',60,255,nothing) #78
cv2.createTrackbar('S1 Higher','Color 1',255,255,nothing) #255
cv2.createTrackbar('V1 Lower','Color 1',200,255,nothing) #128
cv2.createTrackbar('V1 Higher','Color 1',255,255,nothing) #255


while(cap.isOpened()):

    ret,frame=cap.read()
    if ret==True:
        k=cv2.waitKey(1)

    

    #converting to HSV
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hue,sat,val = cv2.split(hsv)

    hmn=cv2.getTrackbarPos('H1 Lower','Color 1')
    smn=cv2.getTrackbarPos('S1 Lower','Color 1')
    vmn=cv2.getTrackbarPos('V1 Lower','Color 1')
    hmx=cv2.getTrackbarPos('H1 Higher','Color 1')
    smx=cv2.getTrackbarPos('S1 Higher','Color 1')
    vmx=cv2.getTrackbarPos('V1 Higher','Color 1')

    # Apply thresholding
    hthresh = cv2.inRange(np.array(hue),np.array(hmn),np.array(hmx))
    sthresh = cv2.inRange(np.array(sat),np.array(smn),np.array(smx))
    vthresh = cv2.inRange(np.array(val),np.array(vmn),np.array(vmx))

    # AND h s and v
    tracking = cv2.bitwise_and(hthresh,cv2.bitwise_and(sthresh,vthresh))

    # Some morpholigical filtering
    erosion=cv2.erode(tracking,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    closing = cv2.GaussianBlur(closing,(5,5),0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(closing,cv.CV_HOUGH_GRADIENT,2,120,param1=120,param2=50,minRadius=10,maxRadius=0)
    # circles = np.uint16(np.around(circles))

    c_centre=[]

    #Draw Circles
    if circles is not None:
            for i in circles[0,:]:
                # If the ball is far, draw it in green
                #if int(round(i[2])) < 30:
                cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,255,0),5)
                cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),2,(0,255,0),10)
                cx1=int(round(i[0]))
                cy1=int(round(i[1]))
                c_centre.append((int(round(i[0])),int(round(i[1]))))



				# else draw it in red
                #elif int(round(i[2])) > 35:
                #    cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),int(round(i[2])),(0,0,255),5)
                #    cv2.circle(frame,(int(round(i[0])),int(round(i[1]))),2,(0,0,255),10)
                #    buzz = 1

	#you can use the 'buzz' variable as a trigger to switch some GPIO lines on Rpi :)
    # print buzz                    
    # if buzz:
        # put your GPIO line here

    #cv2.circle(img,(cx1,cy1),10,(255,0,0),-1)
    #print cx1,cy1
    
    #cv2.circle(frame,(centre_x,centre_y),5,(255,0,0),-1)
    #cv2.rectangle(frame,(300,220),(340,260),(255,0,0),2)
    # if (cx1 > 0) and (cx1 < 320) :
    #     print "Start 1"
    #     ser.write(chr(1))
    #     time.sleep(5)
    #     print "LED 1"
    # elif (cx1 > 320) and (cx1 < 640) :
    #     print "Start 2"
    #     ser.write(chr(2))
    #     time.sleep(5)
    #     print "LED 2"
    # elif (cx1 > 300) and (cx1 < 340) :
    #     print "Start 5"
    #     ser.write(chr(5))
    #     time.sleep(5)
    #     print "LED 5"
    #Show the result in frames
    #cv2.imshow('HueComp',hthresh)
    #cv2.imshow('SatComp',sthresh)
    #cv2.imshow('ValComp',vthresh)
    cv2.imshow('closing',closing)
    cv2.imshow('tracking',tracking)
    cv2.imshow('frame',frame)

    #k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()

for c in range(0,len(c_centre)):
    print "Circle {0} : ".format(c),c_centre[c]

cv2.imshow('frame',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

max_robots=len(c_centre)

kmeans = KMeans(n_clusters=max_robots)
kmeans.fit(pattern_coords)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# print "centroids ",(centroids)
# print "(labels) ",(labels)

colors = ["g.","r.","c.","y."]

for i in range(len(pattern_coords)):
    #print("coordinate:",pattern_coords[i], "label:", labels[i])
    plt.plot(pattern_coords[i][0], pattern_coords[i][1], "g.", markersize = 10)
    #cv2.circle(img,(pattern_coords[i][0], pattern_coords[i][1]), 10, (255,0,0), -1)

# for j in range(len(centroids)):
#     cv2.circle(frame,(int(centroids[j][0]), int(centroids[j][1])), 10, (255,0,0), 3)

#cv2.imshow('clusttered plot',frame)
#plt.scatter(centroids[:, 1],centroids[:, 0], marker = "x", s=150, linewidths = 5, zorder = 10)
#plt.scatter(centroids[:, 1],centroids[:, 0], marker = "x", s=150, linewidths = 5, zorder = 10)

robot_coords_initial=c_centre
C = cdist(robot_coords_initial, centroids)
_, assignment = linear_sum_assignment(C)

#print 'assignment : ',assignment

plt.axis([0,w,h,0])

for p in range(max_robots):
    
    plt.plot(robot_coords_initial[p][0], robot_coords_initial[p][1],'bo', markersize = 10)
    plt.plot(centroids[p][0], centroids[p][1],'rs',  markersize = 7)
    plt.plot((robot_coords_initial[p][0], centroids[assignment[p]][0]), (robot_coords_initial[p][1], centroids[assignment[p]][1]), 'k')
    
    cv2.circle(frame,(robot_coords_initial[p][0], robot_coords_initial[p][1]), 10, (255,0,0), 5)
    cv2.circle(frame,(int(centroids[assignment[p]][0]), int(centroids[assignment[p]][1])), 5, (0,0,255), -1)
    cv2.line(frame,(robot_coords_initial[p][0], robot_coords_initial[p][1]),(int(centroids[assignment[p]][0]), int(centroids[assignment[p]][1])),(0,0,0),2)
#plt.xlim(-1.1,1.1)
#plt.ylim(-1.1,1.1)
plt.axes().set_aspect('equal')

cv2.imshow('clusttered plot',frame)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()