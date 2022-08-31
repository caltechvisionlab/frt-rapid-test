import numpy as np

def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection area
    # and dividing it by the sum of prediction + ground-truth areas - intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def bbox_iou_matrix(boxes):
    # convert the boxes to a numpy array
    boxes = np.array(boxes)
    
    # calculate the number of boxes
    num_boxes = len(boxes)
    
    # create an empty matrix to hold the pairwise IoU values
    iou_matrix = np.zeros((num_boxes, num_boxes))
    
    # calculate the IoU value for each pair of boxes
    for i in range(num_boxes):
        for j in range(i+1, num_boxes):
            iou = bbox_iou(boxes[i], boxes[j])
            iou_matrix[i, j] = iou
            iou_matrix[j, i] = iou
    
    return iou_matrix