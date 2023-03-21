import cv2
import numpy as np
from camera import *

model_list = ["models/sushi/bang_sushi.weights", "models/sushi/cham_sushi.weights", "models/sushi/daeha.weights",
  "models/sushi/doldom_sushi.weights","models/sushi/godeong_sushi.weights","models/sushi/gwang_sushi.weights",
  "models/sushi/jeon_sushi.weights","models/sushi/jeonbok_sushi.weights","models/sushi/squid_sushi.weights",
  "models/sushi/yeolgi_sushi.weights","models/sushi/yeon_sushi.weights"
]
class_list = [["bang_sushi"],["cham_sushi"],["daeha"],["doldom_sushi"],["godeong_sushi"],["gwang_sushi"],["jeon_sushi"],
  ["jeonbok_sushi"],["squid_sushi"],["yeolgi_sushi"],["yeon_sushi"]]
img_list = []
final_sushi_result = []
confidence_sushi_list = []

def detectSushiModels():
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
          # print('confidnece : ', confidence)
          
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
          # final_sushi_result.append(new_result)
          # confidence_sushi_list.append(confidence)
          boxes.append([x, y, w, h])
          confidences.append(float(confidence))
          class_ids.append(class_id)
          
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
      if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence_sushi_list.append(confidences[i])
        final_sushi_result.append(label)
        print(label, ':', confidences[i])
        # print(i, label)
        color = colors[0]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)
    # return cv2.imencode('.jpg', img)
    # print(class_ids, confidences)
    img_list.append(img)
  return img_list


def get_final_sushi_result():
  return final_sushi_result
def clear_final_sushi_result():
  final_sushi_result.clear()
  img_list.clear()
def get_confidence_sushi_list():
  return confidence_sushi_list
def clear_confidence_sushi_list():
  confidence_sushi_list.clear()
def get_best_sushi():
  print('현재 회별 확률 : ',confidence_sushi_list)
  best_confidence = max(confidence_sushi_list)
  index = confidence_sushi_list.index(best_confidence)
  print('확률 높은 회 : ', final_sushi_result[index], confidence_sushi_list[index])
  return final_sushi_result[index], confidence_sushi_list[index]