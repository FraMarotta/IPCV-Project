import cv2
import matplotlib.pyplot as plt
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
#---------------------------------------------------------
files = glob.glob("img/*.bmp")  #all the imgs paths
images = []
for path in files:
    images.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE )) #read and store all the imgs

for k,gray in enumerate(images): 
    print("img ", k)
    if(k>4):
        break  #break after the 1st img until now, poi si toglie sia k che sto if
    #show it
    """ plt.imshow(gray, cmap='gray')
    plt.show() """

    #remove impulsive noise (powder), we may need more than one pass of the filter -> function that does this
    gray = median_filter(gray, 3,3) #remove impulsive noise (powder), we may need more than one pass of the filter -> function that does this
    
    #binarize it by the OTZU's method
    th, binarized_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #show the two images
    """ plt.subplot(1,2,1)
    plt.title("Original image")
    plt.imshow(gray, cmap='gray')

    plt.subplot(1,2,2)
    plt.title("Thresholded by Otsu's algorithm")
    plt.imshow(binarized_image, cmap='gray')
    plt.show() """

    ##### TO DO erosion 

    #label the binarized image with connected components to find rods (and other object). Rule of neighbourhood=4
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(binarized_image, 4)

    #now iter on the found object, the first labeled thing is the bg ->start from 1

    for i in range(1, retval):

        #crete an "object mask", a black(0) img with only the current component in white (255)
        object = np.zeros_like(labels, dtype=np.uint8)
        object[labels==i] = 255

        #in centroids[i] we have the position of each rod (1.2)

        #label the object mask, but we have to invert it beacause opencv look for white things, thus we invert out image to search the holes
        retval_in, labels_in, stats_in, centroids_in = cv2.connectedComponentsWithStats(255 - object, 4)


        if((retval_in-2) > 2): #if we have more than 2 holes we can skip to next object   #after MER implement that under some elongateness we do not have rod even with hole
            continue

        holes = [] #store here holes info (1.4)

        #iter on the holes
        #the first thing is the component on which we are working, the second the real bg -> start from 2
        for j in range(2, retval_in):
            #holes task done (1.4)
            #(centroids_in[j], diameter) are the Holes information (center and diameter)
            diameter = 2 * math.sqrt(stats_in[j][4]/math.pi)
            hole = (centroids_in[j], diameter)
            holes.append(hole)  
            #retval_in-2 = number of holes  (1.1)
        
        #now we have to compute the remaining things
        #-orientation (modulo pi)
        #-Length (L), Width (W), Width at the barycenter (WB)   ##from MER

        #prova sul mer
        cntrs, _= cv2.findContours(object, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        c = cntrs[-1]
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)  #mer da capire cosa contiene
        [vx,vy,x,y] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
        
        angle_rad = -math.atan2(vy, vx)
        angle_degrees = math.degrees(angle_rad)
        if angle_degrees < 0:
            angle_degrees += 180  
        #orientation angle obtained in angle_degrees(1.2)
        """ draw_oriented_mer(binarized_image, box)
        cv2.imshow("prova",binarized_image)
        cv2.moveWindow("prova", 0, 0)
        cv2.waitKey()
        cv2.destroyAllWindows() """
        #print("width:", rect[1][0], "height:", rect[1][1])
        print(stats)
        print('#')

            
        
    