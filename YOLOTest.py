import cv2
import cv2 as cv
import numpy as np
from tracker import *

cap = cv.VideoCapture(r"C:\Users\MSI\Desktop\python\Drone Surveillance Counter - Computer Vision Zone.mp4")
whT = 320
confThreshold = 0.5
nmsThreshold = 0.2
id = []
classNames = ["car"]

## Model Files
modelConfiguration = r"C:\Users\MSI\Desktop\python\car_image\yolo_custom_detection\yolov3_testing.cfg"
modelWeights = r"C:\Users\MSI\Desktop\python\car_image\yolo_custom_detection\yolov3_training_last.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
tracker = EuclideanDistTracker()

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    detections = []
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        #cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
        #cv.putText(img, f'{classNames[classIds[i]].upper()}',
        #           (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        detections.append([x, y, w, h])

    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
        global id
        x, y, w, h, id = box_id
        cv2.putText(img, f'{classNames[classIds[i]].upper()}' + str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)


while True:
    success, img = cap.read()
    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)
    img = cv2.resize(img, (768, 432))
    cv2.putText(img, str(id), (660, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.putText(img, 'THINETH', (340, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
    cv.imshow('Image', img)
    cv.waitKey(1)
