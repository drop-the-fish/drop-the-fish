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
  base64Image = request.json[0]['image']
  imageStr = base64.b64decode(base64Image)
  nparr = np.fromstring(imageStr, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  # img = cv2.imread('static/images/fish/salmon.jpg', cv2.IMREAD_COLOR)
  result = detectFishModels(img_np)
  print('result 길이 : ', len(result))
  for i in range(len(result)):

    cv2.imwrite('./static/images/fish/' + fish_class_list[i][0] + '_result_pic' + '.jpg', result[i])
  temp = get_final_result()
  if len(temp) == 0:
    prediction = '어종 인식에 실패했습니다.'
  else:
    final_result, prediction = get_best_fish()
    prediction_num = int(prediction * 100)
    prediction = final_result + ' 일 확률이 ' + str(prediction_num) + ' %입니다.'

  clear_final_result()
  clear_confidence_list()
  data = {result: prediction}
  return jsonify(data)

# @app.route('/result/sushi')
# def result_sushi():
#   result = detectSushiModels()
#   print("result 길이 : ",result)
#   for i in range(len(result)):
#     ret, jpg = cv2.imencode('.jpg', i)
#     cv2.imwrite('./static/images/sushi/' + sushi_class_list[i][0] + '_result_pic' + '.jpg', result[i])
#   temp = get_final_sushi_result()[0]
#   if len(temp) == 0:
#     final_result = "filter_fail"
#     prediction = '회 인식에 실패했습니다.'
#   else:
#     # final_result = get_final_sushi_result()[0]
#     # print('final result 길이 : ', final_result)
#     # prediction = get_confidence_sushi_list()
#     # print('confidence list : ', prediction)
#     # prediction = max(prediction)
#     final_result, prediction = get_best_sushi()
#     prediction_num = int(prediction * 100)
#     prediction = final_result + ' 일 확률이 ' + str(prediction_num) + ' %입니다.'
#     print('정수 변환 : ', prediction)
#   print('final_result : ', final_result)
#   final_result_path = final_result + '_result_pic' + '.jpg'
#   clear_final_sushi_result()
#   clear_confidence_sushi_list
#   return render_template('result_sushi.html', path = final_result_path, prediction = prediction)

if __name__ == '__main__':
  app.run(debug=True)