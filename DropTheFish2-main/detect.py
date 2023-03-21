# import cv2
# import numpy as np
# from camera import *
# net = cv2.dnn.readNet(f"models/bang-obj_best.weights", "models/yolov4-obj.cfg")



# classes = ['bangeo']
# # with open('models/coco.names', 'r') as f:
# #   classes = [line.strip() for line in f.readlines()]
# print(net)
# layer_names = net.getLayerNames()
# print(layer_names)
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# colors = np.random.uniform(0, 255, size=(len(classes), 3))

# result_pic = ''
# def detectAndDisplay():
#   frame = get_now_frame_for_detect()
#   min_confidence = 0.5
#   img = cv2.resize(frame, None, fx=0.4, fy=0.4)
#   height, width, channels = img.shape
#   blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#   net.setInput(blob)
#   outs = net.forward(output_layers)
#   print(outs)
#   class_ids = []
#   confidences = []
#   boxes = []

#   for out in outs:
#     for detection in out:
#       scores = detection[5:]
#       class_id = np.argmax(scores)
#       confidence = scores[class_id]

#       if confidence > min_confidence:
#         # Object detected
#         center_x = int(detection[0] * width)
#         center_y = int(detection[1] * height)
#         w = int(detection[2] * width)
#         h = int(detection[3] * height)

#         # Rectangle coordinates
#         x = int(center_x - w / 2)
#         y = int(center_y - h / 2)

#         boxes.append([x, y, w, h])
#         confidences.append(float(confidence))
#         class_ids.append(class_id)
        
#   indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
#   font = cv2.FONT_HERSHEY_PLAIN
#   for i in range(len(boxes)):
#     if i in indexes:
#       x, y, w, h = boxes[i]
#       label = str(classes[class_ids[i]])
#       # print(i, label)
#       color = colors[i]
#       cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#       cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)
#   # return cv2.imencode('.jpg', img)
#   # print(class_ids, confidences)
#   return img 

