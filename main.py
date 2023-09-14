# Authors: Francesco Marotta, Simone Maravigna
# Date: 2023-09
# Project: Motorcycle Connecting Rods Inspection - Computer Vision and Image Processing Project Course
# Description: The project aims to develop a system for the automatic inspection of motorcycle connecting rods.

#---------------------------------------------------------
#libraries
#---------------------------------------------------------
import cv2
import numpy as np
import math
import glob

#---------------------------------------------------------
#helper functions
#---------------------------------------------------------
#median filter with n iterations and kernel size
def median_filter(image, n_iters, kernel_size):
    for i in range(n_iters):
        image = cv2.medianBlur(image, kernel_size)
    return image

#show an image in a window
def show_image(image, name="Image"):
    cv2.imshow(name, image)
    cv2.moveWindow(name, 0, 0)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#if width > length -> swap
def find_wl(width, length):
    if width > length:
        return length, width
    else:
        return width, length

#find the orientation angle of the rod    
def find_orientation(vx, vy):
    angle_rad = -math.atan2(vy[0], vx[0]) #angle between the x axis and the line that best fits the contour
    angle_degrees = math.degrees(angle_rad)
    if angle_degrees < 0: #if the angle is negative we add 180 to have it in the range [0, 180]
        angle_degrees += 180 
    return angle_degrees

#print all the requested info of the rods
def print_info(holes, position, angle, length, width, distance):
    if len(holes) == 1:
        print("Rod Type: \tType A")
    elif len(holes) == 2:
        print("Rod Type: \tType B")
    print("Position: \t", round(position[0],2),"," ,round(position[1],2))
    print("Orientation angle: \t", round(angle,2), "degrees")
    print("Dimensions: \tLenth=", round(length,2),";" ,"Width=", round(width,2),";" , "Baricenter width=", round(distance,2))
    for hole in holes:
        print("Hole: \t", round(hole[0][0],2),",", round(hole[0][1],2),";" ,"Diameter=", round(hole[1],2))
    print("\n\n")
    
def find_perpendicular_vector(vx, vy):
    return -vy, vx

def find_line_from_point_and_vector(x, y, vx, vy):
    m = vy[0]/vx[0]
    q = y-m*x
    return m, q

def distance_between_points(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def distance_between_point_and_line(x, y, m, q):
    return abs(m*x - y + q)/math.sqrt(m**2 + 1)

#check if a point is not too close to another one
def is_not_close(x1, y1, points, threshold):
    for i in range(len(points)):
        x2 = points[i][0]
        y2 = points[i][1]
        if distance_between_points(x1, y1, x2, y2) < threshold:
            return False
    return True

#find the intersections between the minor axis and the contour, to find the width at the baricenter
def find_intersections(vx_major, vy_major,x_bar,y_bar, contour):
    vx_minor, vy_minor = find_perpendicular_vector(vx_major, vy_major)
    m_minor, q_minor = find_line_from_point_and_vector(x_bar, y_bar, vx_minor, vy_minor)
    intersections = []
    for i in range(len(contour)):
        x = contour[i][0][0]
        y = contour[i][0][1]
        #if the distance between the point and the line is less than 1 pixel, we have an intersection
        if distance_between_point_and_line(x,y, m_minor, q_minor) < 1 and is_not_close(x, y, intersections, 5):
            intersections.append((x, y))
    return intersections

def divide_touching_components(binary_image):
    while True:

        flag = True
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # we iterate on the contours found, we check if the area of the contour is greater than 6000 (two rods touching)
        for i, cnt in enumerate(contours):

            if abs(cv2.contourArea(cnt, True)) > 6000:
                flag = False
                idx = i # we store the index of the contour that is touching another one
                break

        # no touching components found, we can exit the while loop
        if flag:
            break

        # select the contour that is needed to be divided
        contour = contours[idx]
        # we approximate the contour with a polygon
        hull = cv2.convexHull(contour, returnPoints=False)
        # we find the defects of the contour
        defects = cv2.convexityDefects(contour, hull)

        # contour's channels are indexed by the index found with convexityDefects
        points_info = []  # we store here the points of the contour that are defects and their distance from the convex hull
  
        for dft in defects:
            #defects shape is (n, 1, 4), where n is the number of defects found
            #defects are stored as [start_point, end_point, farthest_point, distance]
            _, _, f, d = dft[0]
            # far is the index of the farthest point in countour array 
            far = contour[f][0]
            points_info.append([far, d]) #d is the distance between the farthest point and the convex hull
            
        points_info = sorted(points_info, key=lambda x: x[1])
        print(points_info[0])
        # we select the two points that are the farthest from the convex hull and compute their distance
        likely_points = [points_info[-1][0], points_info[-2][0]]
        distance = distance_between_points(likely_points[0][0], likely_points[0][1], likely_points[1][0], likely_points[1][1])

        # if the distance is greater than 50, we select the third farthest point
        if distance > 50:
            likely_points[1] = points_info[-3][0]
            distance = distance_between_points(likely_points[0][0], likely_points[0][1], likely_points[1][0], likely_points[1][1])

        # draw a line between the two points
        cv2.line(binarized_image, likely_points[0], likely_points[1], (0, 0, 0), 2)

#---------------------------------------------------------
#main
#---------------------------------------------------------
files = glob.glob("img/*.bmp")  #find all the images paths
images = []
for path in files:
    images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) #read and store all the images

for k, gray in enumerate(images): 
    print("Image: ", files[k])
   
    show_image(gray, "Original Image")  

    #remove impulsive noise (powder) with a median filter (3x3) and 3 iterations
    gray = median_filter(gray, 3, 3)
    
    #binarize it by the OTSU's method
    _ , binarized_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    #divide touching components
    divide_touching_components(binarized_image)

    #rgb image copy to draw on it
    display = cv2.cvtColor(binarized_image.copy(), cv2.COLOR_GRAY2RGB)
    show_image(display, "Binarized Image") 

    #label the binarized image with connected components to find rods (and other objects). Rule of neighbourhood=4
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, 4)
    
    #iter on the found objects, the first labeled component is the background -> start from 1

    for i in range(1, retval):
        #crete an "object mask", a black (0) image with only one component in white (255)
        object = np.zeros_like(labels, dtype=np.uint8)
        object[labels==i] = 255

        #in centroids[i] we have the position of each rod (1.2)

        #label the object mask, but we have to invert it beacause opencv look for white things, thus we invert out image to search the holes
        retval_in, labels_in, stats_in, centroids_in = cv2.connectedComponentsWithStats(255 - object, 4)

        #if we have more than 2 holes or no holes(screw) we can skip to next object
        n_holes = retval_in-2 
        if(n_holes > 2 or n_holes == 0):
            continue

        holes = [] #store here holes info (1.4)

        #iter on the holes
        #the first thing is the component on which we are working, the second is the real background -> start from 2
        for j in range(2, retval_in):
            #holes (1.4)
            #(centroids_in[j], diameter) are the Holes information (center and diameter)
            diameter = 2 * math.sqrt(stats_in[j][4]/math.pi)
            hole = (centroids_in[j], diameter)
            holes.append(hole)
        
        #find Minimum Enclosing Rectangle (MER)
        cntrs, _= cv2.findContours(object, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE)
        c = cntrs[-1]
        rect = cv2.minAreaRect(c)

        #compute length as the extent of the object along th emajor axis, and width along the minor axis
        width, length = find_wl(rect[1][0], rect[1][1])
        elongatedness = length/width

        #if elongatedness < 2 (washer) we can skip to next object
        if elongatedness < 2:
            continue

        #orientation angle (1.2)
        [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
        angle_degrees = find_orientation(vx, vy)

        #find the perpendicular line to the major axis of the MER that passes through the centroid of the object
        intersections = find_intersections(vx, vy, centroids[i][0], centroids[i][1], c)

        #draw width at baricenter
        cv2.circle(display, (intersections[0][0],intersections[0][1]), radius=3, color=(255, 0, 0), thickness=-1)
        cv2.circle(display, (intersections[1][0],intersections[1][1]), radius=3, color=(255, 0, 0), thickness=-1)
        cv2.line(display, (int(intersections[0][0]), int(intersections[0][1])), (int(intersections[1][0]), int(intersections[1][1])), (0,0,255), 2)
        show_image(display)

        #compute the width at the baricenter
        distance = distance_between_points(intersections[0][0], intersections[0][1], intersections[1][0], intersections[1][1])

        #print all the requested info of the rods
        print_info(holes, centroids[i], angle_degrees, length, width, distance)

    print('----------------------------------')