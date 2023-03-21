import cv2
import numpy as np
from camera import *
# net = cv2.dnn.readNet(f"models/bang-obj_best.weights", "models/yolov4-obj.cfg")
# model_list = ["models/fish/bang.weights", "models/fish/cham.weights", "models/fish/galchi.weights",
#   "models/fish/godeong.weights","models/fish/gwang.weights","models/fish/jeonbok.weights",
#   "models/fish/song.weights","models/fish/yeolgi.weights","models/fish/yeon.weights"
# ]
model_list = [ "models/fish/cham.weights" ,"models/fish/godeong.weights"
]
class_list = [["bang"],["cham"],["galchi"],["godeong"],["gwang"],["jeonbok"],["song"],["yeolgi"],["yeon"]]
img_list = []
confidence_list = []
final_result = []

def detectFishModels():
  for k in range(len(model_list)):
    net = cv2.dnn.readNet(model_list[k], "models/yolov4-obj.cfg")
    classes = class_list[k]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[k - 1] for k in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    frame = get_now_frame_for_detect()
    min_confidence = 0.5
    img = cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # print(outs)
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
      for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > min_confidence:
          # Object detected
          center_x = int(detection[0] * width)
          center_y = int(detection[1] * height)
          w = int(detection[2] * width)
          h = int(detection[3] * height)

          # Rectangle coordinates
          x = int(center_x - w / 2)
          y = int(center_y - h / 2)
          # final_result에 추가
          # if len(final_result) > 0:
          #   for q in range(len(final_result)):
          #     # 전에 저장된 confidence와 현재 confidence 비교
          #     pre_confidence = final_result[q]
          #     if confidence > pre_confidence:
          #       new_result = class_list[k]
          #       final_result.pop()
          #       final_result.append(new_result)
          #       print('이전 값을 삭제하고 새 값을 넣습니다.')
          # else:
          #   new_result = class_list[k]
          #   final_result.append(new_result)
          # new_result = class_list[k][0]
          # final_result.append(new_result)
          
          boxes.append([x, y, w, h])
          confidences.append(float(confidence))
          class_ids.append(class_id)
          
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
      if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence_list.append(confidences[i])
        final_result.append(label)
        print(label,':',confidences[i])
        # print(i, label)
        color = colors[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)
    # return cv2.imencode('.jpg', img)
    # print(class_ids, confidences)
    img_list.append(img)
  return img_list


def get_final_result():
  return final_result
def clear_final_result():
  final_result.clear()
  img_list.clear()
def get_confidence_list():
  return confidence_list
def clear_confidence_list():
  confidence_list.clear()
def get_best_fish():
  print('현재 어종별 확률 : ',confidence_list)
  best_confidence = max(confidence_list)
  index = confidence_list.index(best_confidence)
  print('확률 높은 어종 : ', final_result[index], confidence_list[index])
  return final_result[index], confidence_list[index]


