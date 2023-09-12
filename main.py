import cv2
import numpy as np
import math
import glob

#---------------------------------------------------------
def median_filter(image, n_iters, kernel_size):
    for i in range(n_iters):
        image = cv2.medianBlur(image, kernel_size)
    return image

def draw_mer(image, mer):
    v1, v2, v3, v4 = mer[0], mer[1], mer[2], mer[3]

    cv2.line(image, (int(v1[0]), int(v1[1])), (int(v2[0]), int(v2[1])), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (int(v2[0]), int(v2[1])), (int(v3[0]), int(v3[1])), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (int(v3[0]), int(v3[1])), (int(v4[0]), int(v4[1])), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (int(v4[0]), int(v4[1])), (int(v1[0]), int(v1[1])), (255, 255, 255), 1, cv2.LINE_AA)

def show_image(image, name="Image"):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_wl(width, length):
    #if width > length -> swap
    if width > length:
        return length, width
    else:
        return width, length
    
def find_orientation(vx, vy):
    angle_rad = -math.atan2(vy[0], vx[0]) #angle between the x axis and the line that best fits the contour
    angle_degrees = math.degrees(angle_rad)
    if angle_degrees < 0: #if the angle is negative we add 180 to have it in the range [0, 180]
        angle_degrees += 180 
    return angle_degrees

def print_info(holes, position, angle, length, width, distance):
    if len(holes) == 1:
        print("Rod Type:    Type A")
    elif len(holes) == 2:
        print("Rod Type:    Type B")
    print("Position:    Centroid=", round(position[0],2),"," ,round(position[1],2))
    print("Orientation angle:   ", round(angle,2), "degrees")
    print("Dimensions: Lenth=", round(length,2), "Width=", round(width,2))
    print("Baricenter width:", round(distance,2))
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

def is_close(x1, y1, points, threshold):
    for i in range(len(points)):
        x2 = points[i][0]
        y2 = points[i][1]
        if distance_between_points(x1, y1, x2, y2) < threshold:
            return False
    return True

def find_intersections(vx_major, vy_major,x_bar,y_bar, contour):
    vx_minor, vy_minor = find_perpendicular_vector(vx_major, vy_major)
    m_minor, q_minor = find_line_from_point_and_vector(x_bar, y_bar, vx_minor, vy_minor)
    intersections = []
    for i in range(len(contour)):
        x = contour[i][0][0]
        y = contour[i][0][1]
        if distance_between_point_and_line(x,y, m_minor, q_minor) < 1 and is_close(x, y, intersections, 5):
            intersections.append((x, y))
    return intersections, q_minor

def divide_connected_components(image, display):
    # using watershed algorithm
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations = 3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    show_image(sure_fg, "Sure Foreground")
    show_image(sure_bg, "Sure Background")
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    markers = cv2.watershed(display, markers)
    display[markers == -1] = 0
    show_image(display, "Watershed")
    return display


#---------------------------------------------------------
files = glob.glob("img/*.bmp")  #find all the images paths
images = []
for path in files:
    images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) #read and store all the images

for k, gray in enumerate(images): 
    print("Image: ", files[k])
    if(k>20):
        break  #break after the 1st img until now, poi si toglie sia k che sto if

    
    #show_image(gray, "Original Image")  #uncomment to show the image

    #remove impulsive noise (powder), we may need more than one pass of the filter -> function that does this
    gray = median_filter(gray, 3, 3)
    
    #binarize it by the OTSU's method
    th, binarized_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    display = cv2.cvtColor(binarized_image.copy(), cv2.COLOR_GRAY2RGB)
    #show_image(binarized_image, "Binarized Image") #uncomment to show binarized image

    ##### TO DO erosion
    divided = divide_connected_components(binarized_image, display)

    #label the binarized image with connected components to find rods (and other objects). Rule of neighbourhood=4
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, 4)
    #iter on the found objects, the first labeled component is the bg -> start from 1

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
        #the first thing is the component on which we are working, the second is the real bg -> start from 2
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
        box = cv2.boxPoints(rect) #find the 4 vertices of the MER

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
        intersections, q_minor = find_intersections(vx, vy, centroids[i][0], centroids[i][1], c)

        #cv2.line(display, (int(0), int(q_minor)), (int(centroids[i][0]), int(centroids[i][1])), (0,255,0), 2)
        #cv2.circle(display, (intersections[0][0],intersections[0][1]), radius=3, color=(0, 0, 255), thickness=-1)
        #cv2.circle(display, (intersections[1][0],intersections[1][1]), radius=3, color=(0, 0, 255), thickness=-1)
        #cv2.drawContours(display, c, -1, (255, 0, 0), 2)
        #cv2.line(display, (int(intersections[0][0]), int(intersections[0][1])), (int(intersections[1][0]), int(intersections[1][1])), (0,255,0), 2)
        #show_image(display)
        distance = distance_between_points(intersections[0][0], intersections[0][1], intersections[1][0], intersections[1][1])
        """ draw_mer(binarized_image, box)
        show_image(binarized_image, "MER") """
        print_info(holes, centroids[i], angle_degrees, length, width, distance)

    print('----------------------------------')

            
        
    