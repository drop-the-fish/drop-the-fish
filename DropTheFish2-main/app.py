from flask import Flask, render_template, request, Response, redirect, url_for
from camera import *
from detect import *
from detectFish import *
from keras.models import load_model
from keras.preprocessing import image
from detectFish import *
from detectSushi import *
app = Flask(__name__)

fish_class_list = [["bangeo"],["cham"],["galchi"],["godeong"],["gwang"],["jeonbok"],["song"],["yeolgi"],["yeon"]]
sushi_class_list = [["bang_sushi"],["cham_sushi"],["daeha"],["doldom_sushi"],["godeong_sushi"],["gwang_sushi"],["jeon_sushi"],
  ["jeonbok_sushi"],["squid_sushi"],["yeolgi_sushi"],["yeon_sushi"]
]


@app.route('/')
def index():
  return render_template('index.html')
  
@app.route('/fish')
def fish():
  return render_template('fish.html')
def gen(camera, type):
  if type == 'video':
    print('녹화를 시작합니다.')
    while True:
      frame = camera.get_frame()
      if frame == 'done':
        break
      yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      
  elif type == 'capture':
    print('capture를 시작합니다.')
    frame = camera.get_frame()
    yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/sushi')
def sushi():
  return render_template('sushi.html')
def gen(camera, type):
  if type == 'video':
    print('녹화를 시작합니다.')
    while True:
      frame = camera.get_frame()
      if frame == 'done':
        break
      yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
      
  elif type == 'capture':
    print('capture를 시작합니다.')
    frame = camera.get_frame()
    yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
  return Response(gen(Video(type='video'), 'video'),
  mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
  return get_now_frame()

@app.route('/result/fish')
def result_fish():
  result = detectFishModels()
  print('result 길이 : ', len(result))
  for i in range(len(result)):
    ret, jpg = cv2.imencode('.jpg', i)
    cv2.imwrite('./static/images/fish/' + fish_class_list[i][0] + '_result_pic' + '.jpg', result[i])
  # ret, jpg = cv2.imencode('.jpg', result)
  # cv2.imwrite('./static/images/' + 'result_pic' + '.jpg', result)
  # return render_template('result.html', prediction = jpg)
  temp = get_final_result()
  if len(temp) == 0:
    final_result = "filter_fail"
    prediction = '어종 인식에 실패했습니다.'
  else:
    # final_result = get_final_result()[0]
    # print('final result 길이 : ', final_result)
    # prediction = get_confidence_list()
    # print('confidence list : ', prediction)
    # prediction = max(prediction)
    final_result, prediction = get_best_fish()
    prediction_num = int(prediction * 100)
    prediction = final_result + ' 일 확률이 ' + str(prediction_num) + ' %입니다.'
    print('정수 변환 : ', prediction)
  print('final_result : ', final_result)
  final_result_path = final_result + '_result_pic' + '.jpg'
  clear_final_result()
  clear_confidence_list()
  return render_template('result_fish.html', path = final_result_path, prediction = prediction)

@app.route('/result/sushi')
def result_sushi():
  result = detectSushiModels()
  print("result 길이 : ",result)
  for i in range(len(result)):
    ret, jpg = cv2.imencode('.jpg', i)
    cv2.imwrite('./static/images/sushi/' + sushi_class_list[i][0] + '_result_pic' + '.jpg', result[i])
  temp = get_final_sushi_result()[0]
  if len(temp) == 0:
    final_result = "filter_fail"
    prediction = '회 인식에 실패했습니다.'
  else:
    # final_result = get_final_sushi_result()[0]
    # print('final result 길이 : ', final_result)
    # prediction = get_confidence_sushi_list()
    # print('confidence list : ', prediction)
    # prediction = max(prediction)
    final_result, prediction = get_best_sushi()
    prediction_num = int(prediction * 100)
    prediction = final_result + ' 일 확률이 ' + str(prediction_num) + ' %입니다.'
    print('정수 변환 : ', prediction)
  print('final_result : ', final_result)
  final_result_path = final_result + '_result_pic' + '.jpg'
  clear_final_sushi_result()
  clear_confidence_sushi_list
  return render_template('result_sushi.html', path = final_result_path, prediction = prediction)

if __name__ == '__main__':
  app.run(debug=True)