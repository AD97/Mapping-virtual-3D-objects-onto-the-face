# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:02:20 2019

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 21:17:14 2019

@author: Admin
"""

from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import math
import time
import dlib
import cv2

# load our serialized model from disk
print("[INFO] loading model...")
#net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        
                        ])
framethresh=5
framecount=-1
nose_arr=np.zeros((framethresh,2),dtype=float)
chin_arr=np.zeros((framethresh,2),dtype=float)
leye_arr=np.full((framethresh,2),150,dtype=float)
reye_arr=np.full((framethresh,2),150,dtype=float)
lmouth_arr=np.zeros((framethresh,2),dtype=float)
rmouth_arr=np.zeros((framethresh,2),dtype=float)
dx=0
dy=0
dz=0
actualeyelength=100

filter_sun = cv2.imread("sunglasses.png",cv2.IMREAD_UNCHANGED)
blank_image = np.zeros((184,184,4), np.uint8)
rows,cols,channels = filter_sun.shape
blank_image[61:61+rows,0:cols]=filter_sun
#cv2.imshow("over",blank_image)
filter_m= cv2.imread("moustache.png",cv2.IMREAD_UNCHANGED)
rows1,cols1,channels1 = filter_m.shape

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    frame = vs.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500)
    size=frame.shape
    framecount=framecount+1
    if framecount == framethresh:
        actualeyelength=int((np.mean(leye_arr,axis=0)[0]-np.mean(reye_arr,axis=0)[0]))
	# grab the frame dimensions and convert it to a blob
	
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob through the network and obtain the detections and
	# predictions
	
    net.setInput(blob)
    detections = net.forward()
    (x,y)=(0,0)
    (x2,y2)=(0,0)
	# loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.8:
            continue

		
        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        rect = dlib.rectangle(startX-10,startY-10,endX+10,endY+10)
        shape = predictor(frame,rect)
        shape = face_utils.shape_to_np(shape)
        (x,y)=shape[30]
        nose_arr[framecount%framethresh]=[x,y]
        (x,y)=shape[8]
        chin_arr[framecount%framethresh]=[x,y]
        (x,y)=shape[45]
        leye_arr[framecount%framethresh]=[x,y]
        (x,y)=shape[36]
        reye_arr[framecount%framethresh]=[x,y]
        (x,y)=shape[54]
        lmouth_arr[framecount%framethresh]=[x,y]
        (x,y)=shape[48]
        rmouth_arr[framecount%framethresh]=[x,y]
        if framecount < framethresh:
            continue
        image_points = np.array([
                            (np.mean(nose_arr,axis=0)[0],np.mean(nose_arr,axis=0)[1]),# Nose tip
                            (np.mean(chin_arr,axis=0)[0],np.mean(chin_arr,axis=0)[1]),# Chin
                            (np.mean(leye_arr,axis=0)[0],np.mean(leye_arr,axis=0)[1]),# Left eye left corner
                            (np.mean(reye_arr,axis=0)[0],np.mean(reye_arr,axis=0)[1]),# Right eye right corne
                            (np.mean(lmouth_arr,axis=0)[0],np.mean(lmouth_arr,axis=0)[1]),# Left Mouth corner
                            (np.mean(rmouth_arr,axis=0)[0],np.mean(rmouth_arr,axis=0)[1])# Right mouth corner
                        ], dtype="double")
        
        #for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            #for (x, y) in shape[i:j]:
                #print(name + ":" + "i=" + str(i) + ",j=" + str(j) + ",x=" + str(x) + ",y=" + str(y) )
            #    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            #for cur in range(i,j):
            #    if cur == 39:
            #        (x1,y1)=shape[cur]
                    #print(name + ":" + "i=" + str(cur) + ",x=" + str(x1) + ",y=" + str(y1) )
            #        cv2.circle(frame, (x1, y1), 1, (0, 0, 255), -1)
            #    elif cur == 42:
            #        (x2,y2)=shape[cur]
                    #print(name + ":" + "i=" + str(cur) + ",x=" + str(x2) + ",y=" + str(y2) )
            #        cv2.circle(frame, (x2, y2), 1, (0, 0, 255), -1)
                    
        #opp=0
        #adj=0
        #if y1>y2:
        #    opp=y1-y2
        #    adj=x2-x1
        #elif y1<y2:
        #    opp=y2-y1
        #    adj=x2-x1
        #print("opp = "+str(opp)+"adj = "+str(adj))
        #angle=math.degrees(math.atan(opp/adj))
        #print(str(angle))
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                 [[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype = "double"
                                  )

        print("Camera Matrix :\n {0}".format(camera_matrix))

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        R,_ = cv2.Rodrigues(rotation_vector)
        sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(R[2,1],R[2,2])
            y = math.atan2(-R[2,0],sy)
            z = math.atan2(R[1,0],R[0,0])
        else:
            x = math.atan2(-R[1,2],R[1,1])
            y = math.atan2(-R[2,0],sy)
            z = 0
        rtheta = x
        rphi = y
        rgamma = z
        print ("x = " + str(math.degrees(x)))
        print ("y = " + str(math.degrees(y)))
        print ("z = " + str(math.degrees(z)))
        #print("Rotation Vector X axis:\n {0}".format(math.degrees(rotation_vector[0])))
        #print("Rotation Vector Y axis:\n {0}".format(math.degrees(rotation_vector[1])))
        #print("Rotation Vector Z axis:\n {0}".format(math.degrees(rotation_vector[2])))
        print("Translation Vector:\n {0}".format(translation_vector))
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
        

        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255,255,255), 2)
        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        #cv2.rectangle(frame, (startX, startY), (endX, endY),
                      #(0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
        
        
        rows,cols,channels = blank_image.shape
        d = np.sqrt(rows**2+cols**2)
        focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz=focal
        w = cols
        h = rows
        f = focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(rtheta), -np.sin(rtheta), 0],
                        [0, np.sin(rtheta), np.cos(rtheta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(rphi), 0, -np.sin(rphi), 0],
                        [0, 1, 0, 0],
                        [np.sin(rphi), 0, np.cos(rphi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(rgamma), -np.sin(rgamma), 0, 0],
                        [np.sin(rgamma), np.cos(rgamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        mat = np.dot(A2, np.dot(T, np.dot(R, A1)))
        
        rows1,cols1,channels1 = filter_m.shape
        d = np.sqrt(rows1**2+cols1**2)
        focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
        dz=focal
        w = cols1
        h = rows1
        f = focal

        # Projection 2D -> 3D matrix
        A1 = np.array([ [1, 0, -w/2],
                        [0, 1, -h/2],
                        [0, 0, 1],
                        [0, 0, 1]])
        
        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([ [1, 0, 0, 0],
                        [0, np.cos(rtheta), -np.sin(rtheta), 0],
                        [0, np.sin(rtheta), np.cos(rtheta), 0],
                        [0, 0, 0, 1]])
        
        RY = np.array([ [np.cos(rphi), 0, -np.sin(rphi), 0],
                        [0, 1, 0, 0],
                        [np.sin(rphi), 0, np.cos(rphi), 0],
                        [0, 0, 0, 1]])
        
        RZ = np.array([ [np.cos(rgamma), -np.sin(rgamma), 0, 0],
                        [np.sin(rgamma), np.cos(rgamma), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([  [1, 0, 0, dx],
                        [0, 1, 0, dy],
                        [0, 0, 1, dz],
                        [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([ [f, 0, w/2, 0],
                        [0, f, h/2, 0],
                        [0, 0, 1, 0]])

        # Final transformation matrix
        mat1 = np.dot(A2, np.dot(T, np.dot(R, A1)))
        #(x1,y1)=(np.mean(nose_arr,axis=0)[0],np.mean(nose_arr,axis=0)[1])
        #(x2,y2)=(np.mean(chin_arr,axis=0)[0],np.mean(chin_arr,axis=0)[1])
        #opp=0
        #adj=0
        #opp=x1-x2
        #adj=y2-y1
        #print("opp = "+str(opp)+"adj = "+str(adj))
        #angle=math.atan(opp/adj)
        #print(delta)
        filter_sun1 = cv2.warpPerspective(blank_image, mat, (cols, rows))
        filter_m1 = cv2.warpPerspective(filter_m,mat1,(cols1,rows1))
        rows,cols,channels = filter_sun1.shape
        rows1,cols1,channels1 = filter_m1.shape
        #print(cols)
        #print(rows)
        eyelength = int((np.mean(leye_arr,axis=0)[0]-np.mean(reye_arr,axis=0)[0]))
        multiplier=eyelength/actualeyelength
        sunglass_width=int(cols*multiplier)
        sunglass_height=int(rows*multiplier)
        m_width = int((np.mean(lmouth_arr,axis=0)[0]-np.mean(rmouth_arr,axis=0)[0])*1.4)
        multiplier1 = m_width/cols1
        m_height=int(rows1*multiplier1)
        print(multiplier)
        startx=int(((np.mean(leye_arr,axis=0)[0]+np.mean(reye_arr,axis=0)[0])/2)-((sunglass_width)/2))
        starty=int(np.mean(reye_arr,axis=0)[1])
        startx1=int(((np.mean(leye_arr,axis=0)[0]+np.mean(reye_arr,axis=0)[0])/2)-(m_width/2))
        starty1=int(np.mean(rmouth_arr,axis=0)[1])
        print(startx)
        #print(sunglass_width)
        #multiplier = sunglass_width/cols
        #print(multiplier)
        #sunglass_height = int(multiplier*rows)
        
        #print(sunglass_height)
        #delta = int(math.tan(angle)*(sunglass_width/2))
        if (starty-int(sunglass_height/2))<0:
            extra = int(sunglass_height/2)-(starty)
        else:
            extra = 0
        if (starty1-int(m_height/2))<0:
            extra1 = int(m_height/2) -(starty1)
        else:
            extra1 = 0
        print(extra)
        resized_sun = cv2.resize(filter_sun1,(sunglass_width,sunglass_height),interpolation=cv2.INTER_CUBIC)
        resized_m = cv2.resize(filter_m1,(m_width,m_height),interpolation=cv2.INTER_CUBIC)
        transparent_region = resized_sun[:,:,:3] != 0
        transparent_region1 = resized_m[:,:,:3] != 0
        frame[starty+extra-int(sunglass_height/2):starty+extra+int(sunglass_height/2),startx:startx+sunglass_width][transparent_region] = resized_sun[:,:,:3][transparent_region]
        frame[starty1-10+extra1-int(m_height/2):starty1-10+extra1+int(m_height/2),startx1:startx1+m_width][transparent_region1] = resized_m[:,:,:3][transparent_region1]

	# show the output frame
	
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	
    if key == ord("q"):
        break

# do a bit of cleanup
vs.stop()
cv2.destroyAllWindows()
#vs.stop()