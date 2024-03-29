import cv2
import numpy as np

model_list = ["/home/kohjunghoon/mysite/fish/salmon.weights", "/home/kohjunghoon/mysite/fish/flatfish.weights"]
class_list = [["연어"], ["광어"]]

fish_confidence_list = []
fish_final_result = []


def detectFishModels(img):
    for k in range(len(model_list)):
        net = cv2.dnn.readNet(model_list[k], "/home/kohjunghoon/mysite/yolov4.cfg")
        classes = class_list[k]
        layer_names = net.getLayerNames()
        output_layers = [layer_names[k - 1] for k in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))

        min_confidence = 0.5
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

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

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                fish_confidence_list.append(confidences[i])
                fish_final_result.append(label)
                print(label, ':', confidences[i])

                color = colors[0]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 2, (0, 255, 0), 1)
    return


def get_final_result():
    return fish_final_result


def clear_final_result():
    fish_final_result.clear()


def get_confidence_list():
    return fish_confidence_list


def clear_confidence_list():
    fish_confidence_list.clear()


def get_best_fish():
    print('현재 어종별 확률 : ', fish_confidence_list)
    print('현재 어종별 모델 : ', fish_final_result)
    # return confidence_list
    best_confidence = max(fish_confidence_list)
    index = fish_confidence_list.index(best_confidence)
    print('확률 높은 어종 : ', fish_final_result[index], fish_confidence_list[index])
    return fish_final_result[index], fish_confidence_list[index]
