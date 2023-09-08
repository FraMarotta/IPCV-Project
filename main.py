import cv2
import numpy as np
import math
import glob

#---------------------------------------------------------
def median_filter(image, n_iters, kernel_size):
    for i in range(n_iters):
        image = cv2.medianBlur(image, kernel_size)
    return image

def draw_oriented_mer(image, mer):
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

def print_info(holes, position, angle, length, width):
    if len(holes) == 1:
        print("Type A")
    elif len(holes) == 2:
        print("Type B")
    
#---------------------------------------------------------
files = glob.glob("img/*.bmp")  #find all the images paths
images = []
for path in files:
    images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE)) #read and store all the images

for k, gray in enumerate(images): 
    print("Image: ", files[k])
    if(k>4):
        break  #break after the 1st img until now, poi si toglie sia k che sto if
    #show it
    #show_image(gray, "Original Image")

    #remove impulsive noise (powder), we may need more than one pass of the filter -> function that does this
    gray = median_filter(gray, 3, 3)
    
    #binarize it by the OTSU's method
    th, binarized_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #show binarized image
    #show_image(binarized_image, "Binarized Image")

    ##### TO DO erosion

    #label the binarized image with connected components to find rods (and other objects). Rule of neighbourhood=4
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, 4)
    #iter on the found object, the first labeled component is the bg -> start from 1

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
        
        #compute the remaining things
        #-orientation (modulo pi)
        #-Length (L), Width (W), Width at the barycenter (WB)

        #find Minimum Enclosing Rectangle (MER)
        cntrs, _= cv2.findContours(object, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        c = cntrs[-1]
        rect = cv2.minAreaRect(c)

        width, length = find_wl(rect[1][0], rect[1][1])
        elongatedness = length/width
        #if elongatedness < 2 (washer) we can skip to next object
        if elongatedness < 2:
            continue

        box = cv2.boxPoints(rect) #find the 4 vertices of the MER
        [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
        
        #orientation angle (1.2)
        angle_degrees = find_orientation(vx, vy)
        
        """ draw_oriented_mer(binarized_image, box)
        show_image(binarized_image, "MER") """
        print_info(holes, centroids[i], angle_degrees, length, width)

    print('----------------------------------')

            
        
    