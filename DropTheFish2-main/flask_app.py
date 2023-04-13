import json
from flask import Flask, render_template, request, jsonify
from detectFish import *
from detectSushi import *
app = Flask(__name__)
import base64
fish_class_list = [["cham"],["yeon"]]
sushi_class_list = [["cham_sushi"],["salmonsashimi_sushi"],]


@app.route('/')
def index():
  return render_template('index.html')
  
@app.route('/fish')
def fish():
  return render_template('fish.html')

@app.route('/sushi')
def sushi():
  return render_template('sushi.html')

@app.route('/result/fish', methods=['POST', 'GET'])
def result_fish():
  base64Image = request.json
  imageStr = base64.b64decode(base64Image['image'])
  nparr = np.fromstring(imageStr, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  detectFishModels(img_np)

  temp = get_final_result()
  if len(temp) == 0:
      prediction = '어종 인식에 실패했습니다.'
  else:
    final_result, prediction = get_best_fish()
    prediction_num = int(prediction * 100)
    prediction = prediction_num

  clear_final_result()
  clear_confidence_list()
  return jsonify({final_result: str(prediction)}),200

@app.route('/result/sushi', methods=['POST', 'GET'])
def result_sushi():
  base64Image = request.json
  imageStr = base64.b64decode(base64Image['image'])
  nparr = np.fromstring(imageStr, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  detectSushiModels(img_np)

  temp = get_final_result()
  if len(temp) == 0:
      prediction = '어종 인식에 실패했습니다.'
  else:
    final_result, prediction = get_best_fish()
    prediction_num = int(prediction * 100)
    prediction = prediction_num

  clear_final_result()
  clear_confidence_list()
  return jsonify({final_result: str(prediction)}),200



if __name__ == '__main__':
  app.run(debug=True)