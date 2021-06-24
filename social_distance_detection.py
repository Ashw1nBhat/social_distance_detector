import cv2
import numpy as np

#saving all class names in a list
classes = []
with open('data/coco.names','r') as f:
    classes = f.read().splitlines()

#getting the weights file and config file and building the network using dnn module
modelConfig = 'config/yolov3.cfg'
modelWeights = 'weights/yolov3.weights'
net = cv2.dnn.readNet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


#opening the videostream
cap = cv2.VideoCapture("video/video.mp4")
width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
size = (width,height)

#setting up the output videostream
out = cv2.VideoWriter('sample_output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

#getting the first 2 frames
_, frame1 = cap.read()
_, frame2 = cap.read()


def get_all_boxes(layer_outputs):
    """

    get_all_boxes(list[list[float]])

        This method takes all of the predictions made by the network and returns those predictions that have
    confidence value of over 0.5 and class_id of 0.

    output : all_boxes - list of bounding boxes each with (x, y, w, h) values
            confidences - list of confidence values for each of the bounding boxes detected
    """
    all_boxes = []
    confidences = [] 

    for output in layer_outputs:
        for detection in output:

            scores=detection[5:]                
            class_id=np.argmax(scores)          
            confidence =scores[class_id]

            if confidence > 0.5 and class_id==0:

                center_x=int(detection[0]*width)
                center_y=int(detection[1]*height)
                w=int(detection[2]*width)
                h=int(detection[3]*height)

                x=int(center_x-w/2) 
                y=int(center_y-h/2)

                all_boxes.append([x,y,w,h])
                confidences.append(float(confidence))

    return all_boxes, confidences


def get_bounding_boxes(frame):
    """

    get_bounding_boxes(image)

        This method takes an image as input, converts it into a blob which is suitable for passing to the 
    neural network as input. Returns the indices of normalised boxes, confidence values and (x, y, w, h)
    values of all the bounding boxes.

    output : indexes - list of indices of normalised bounding boxes
            confidences - list of confidence values for each of the bounding boxes detected
            all_boxes - list of bounding boxes each with (x, y, w, h) values
    """

    blob = cv2.dnn.blobFromImage(frame,1/255,(320,320),(0,0,0),1,crop=False)
    net.setInput(blob)

    output_layer_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layer_names)

    all_boxes, confidences = get_all_boxes(layer_outputs)

    indexes=cv2.dnn.NMSBoxes(all_boxes,confidences,0.5,0.3)

    return indexes, confidences, all_boxes



def mid_point(img, box):
    """

    mid_point(image, list[float])

        This method takes the coordinates of a box, finds the midpoint of the bottom line and draws a
    circle on that point. It returns the midpoint coordinate.

    output : mid - coordinate of the midpoint (x,y)
    """

    x1, y1, w, h = box[0], box[1], box[2], box[3]
    x2, y2 = x1+w, y1+h
  
    x_mid = int((x1+x2)/2)
    y_mid = int(y2)
    mid = (x_mid,y_mid)
  
    _ = cv2.circle(img, mid, 5, (255, 0, 0), -1)
  
    return mid


def draw_boxes(indexes, frame, all_boxes):
    """

    draw_boxes(list[list[int]], image, list[list[float]])

        This method takes the indices of normalised boxes, the list of boxes themselves and draws a 
    blue rectangle over the normalised ones. Returns the midpoints and a list of (x, y, w, h) values
    of normalised boxes.

    output : mid_points - list of all midpoints
            bbox - list of (x, y, w, h) values of normalised boxes
    """
    bbox = []
    mid_points = []

    for i in indexes:
        x = i[0]
        box = all_boxes[x]
        bbox.append(box)
        mid_points.append(mid_point(frame, box))
        x1, y1, w, h = box[0], box[1], box[2], box[3]
        x2, y2 = x1+w, y1+h

        cv2.rectangle(frame, (x1,y1),(x2,y2),(255,0,0),2) 

    return mid_points, bbox

def compute_distance(point_1, point_2):
    """

    compute_distance(list[int], list[int])

        This method takes the coordinates of 2 points and calculates the Euclidean distance between
    them.

    output : distance - the Euclidean distance between 2 points
    """
    x1, y1, x2, y2 = point_1[0], point_1[1], point_2[0], point_2[1]
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    return distance

def get_distances_list(mid_points):
    """

    get_distances_list(list[float])

        This method takes the midpoints as the input and gives a 2d list which contains the distances 
    between all the boxes, with the index of the list indicating the box number. 

    output : dist_list - a 2d list containing distances
    """
    n = len(mid_points)
    dist_list = np.zeros((n,n))

    for i in range(n):
        for j in range(i+1, n):
            dist_list[i][j] = compute_distance(mid_points[i], mid_points[j])
    
    return dist_list


def find_closest(distances, threshold):
    """

    find_closest(list[float][float], int)

        This method takes the threshold and the distances list, traverses through the list and finds those 
    boxes which are violating the threshold value.

    output : person_1 - list of 1st set of boxes violating the threshold
            person_2 - list of 2nd set of boxes violating the threshold
            d - list of the distances which are in violation
    """
    n = len(distances)
    person_1 = []
    person_2 = []
    d = []

    for i in range(n):
        for j in range(i+1, n):
            if distances[i][j] <= threshold:
                person_1.append(i)
                person_2.append(j)
                d.append(distances[i][j])

    return person_1, person_2, d


def change_bbox_color(img, boxes, p1, p2):
    """

    change_bbox_color(image, list[list[float]], list[int], list[int])

        This method takes the list which contains the indices of violating boxes and draws a red rectangle over 
    them.

    output : img - resulting image
    """
    points = np.unique(p1 + p2)

    for i in points:
        x1, y1, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        x2, y2 = x1+w, y1+h
        _ = cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)  

    return img

#threshold value
threshold = 100


#traversing each frame until the end of the video is reached or ESC key is pressed
while cap.isOpened():

    indexes, confidences, all_boxes = get_bounding_boxes(frame1)
    mid_points, bounding_boxes = draw_boxes(indexes, frame1, all_boxes)
    distances_list = get_distances_list(mid_points)
    p1,p2,d = find_closest(distances_list,threshold)
    img = change_bbox_color(frame1, bounding_boxes, p1, p2)

    out.write(img)
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break


cv2.destroyAllWindows()
cap.release()
out.release()

            


