import cv2
import numpy as np
from numpy.lib.twodim_base import mask_indices

video = cv2.VideoCapture("lane_detection_video.mp4")

def draw_lines(image,lines):
    # set up the picture to draw lines, same size as the image(height,width,channel),8bit per channel
    canvas = np.zeros((image.shape[0],image.shape[1],3),np.uint8)

    # draw the lines on our canvas (src,pt1,pt2,BGR,thinkness)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(canvas,(x1,y1),(x2,y2),(0,255,0),thickness = 3)
    
    # blend two picture together, (src1,alpha_weight,src2,beta_weight,gamma_brightness)
    image_with_lines = cv2.addWeighted(image,0.9,canvas,1,1.0)
    return image_with_lines



def crop_region(frame,region):
    # create a black image
    mask = np.zeros_like(frame)
    # fill the area of interest with white, region need to be int32
    cv2.fillPoly(mask,region,255)
    # bitwise AND operation to keep the aera of interest
    masked_image = cv2.bitwise_and(frame,mask)
    return masked_image

def lane_detection(frame):
    (height,width) = (frame.shape[0],frame.shape[1])
    # Always cvt image to grayscale
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Canny image 1.Gaussian Blur remove noise. 2.Sober Kernel to find the change in intensity or edges, both X and Y direction 
    # 3.sqrt of sum of squared Gx Gy get the average intensity 4. non_maxmium suppresion, anything lower than the lower threshold would gone
    # anything that greater than higher threshold would be kept, anything between would only be kept if it connect to a pixel that is above higher bound.
    canny_image = cv2.Canny(gray_image,100,130)
    # area of interst, a triangle aera
    region = [
        (0,height),
        (width/2,0.6 * height),
        (width,height)
    ]
    masked_image = crop_region(canny_image,np.array([region],np.int32))
    # Hough Transform, caucasian to polar coordinate, get the highest vote of intersections as lines. 
    lines = cv2.HoughLinesP(masked_image,rho = 1,theta = np.pi/180, threshold = 60, minLineLength = 100, maxLineGap = 150)
    image_with_lines = draw_lines(frame,lines)

    return image_with_lines


 
while video.isOpened():
    # Check if we are at last frame and get the current frame
    grabbed,frame = video.read()
    
    # last frame break
    if not grabbed:
        break
    
    # call the lane detection video 
    frame = lane_detection(frame)
    cv2.imshow("lane_detection",frame)
    cv2.waitKey(20)
# clean up
video.release()
cv2.destroyAllWindows()
