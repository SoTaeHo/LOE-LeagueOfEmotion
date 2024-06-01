# LOE - League Of Emotion

LOE - League Of Emotion은 FER(Face Emotion Recognition)을 이용하여 누가 더 감정을 생생하게 표현하는지 겨뤄보는 게임이다.<br/>
감정은 총 7가지(angry, disgust, fear, happy, neutral, sad, surprise)가 있으며, 게임 시작 시 어떤 감정으로 겨룰지 선택할 수 있다.<br/><br/>
감정을 선택하면 Player 1부터 10초동안 웹캠을 바라보고 선택한 감정이 잘 드러나게 표정을 지으면 된다. 10초동안 Player 1의 얼굴을 감지하여 해당 얼굴이 나타내는 감정의 점수를 실시간으로 보여주고 기록한다. Player 1의 차례가 끝나면 Player 2의 차례가 시작되며, 동일하게 10초동안 웹캠을 향해 표정을 지으면 감정의 점수를 측정한다. Player 2의 측정이 끝나면, 10초 동안의 감정의 평균을 구하여 결과창에서 누가 승리하였는지 보여준다.

## 목차
- [특징](#특징)
- [요구 사항](#요구-사항)
- [모델 학습](#모델-학습)
- [게임 플레이](#게임-플레이)

## 요구 사항
- Python 3.10
- OpenCV
- NumPy
- Matplotlib
- tensorflow
- keras

## 모델 학습

생성된 모델의 weight는 emotion_model.h5이며 model_fitting_basic.py를 통해 학습했다. papers wiht code에도 FER2013을 사용하여 감정 인식을 하는 여러 코드들이 있었지만, 해당 코드를 이용하여 어떻게 학습해야하는지 방법을 모르겠어 시도해보지 못했다.<br/>
그 외에도 chatGPT를 이용하여 모델의 성능을 개선할 수 있는 여러 방법(전이 학습, 데이터 전처리, 복잡한 모델 생성, 앙상블 메소드 사용)을 시도해보았지만, 학습 시간이 매우 오래걸리거나 Accuracy같은 성능이 눈에 띄게 향상되지 않아 적당한 성능과 학습 시간을 보유한 해당 모델을 사용하게 되었다.

## 게임 플레이

<div align="center">
  <img src="https://github.com/SoTaeHo/LOE-LeagueOfEmotion/assets/91146046/09facb5c-5795-46fb-96a6-cd643c1283a3" alt="image" width="500"/><br>-게임 시작 화면-
</div><br/>
게임 시작 시 7개의 감정 중 1가지를 선택할 수 있다. 
<div align="center">
<img width="500" alt="player2" src="https://github.com/SoTaeHo/LOE-LeagueOfEmotion/assets/91146046/3f83cfbc-62c8-4151-8187-041ee66ed363"><br>-게임 중 화면을 감지하는 장면-
</div><br/>
게임이 시작되면 10초간 웹캠을 통해 선택한 감정이 최대한 드러나게 표정을 지으면 된다. 측정되는 감정은 실시간으로 변하며, 감정들은 0 ~ 1사이의 값으로 표현되며, 모든 감정의 총합은 1이다. 예를 들어 화난 표정을 짓고 싶으면 미간을 찡그리거나, 눈을 크게 뜨면 된다. 반대로 행복한 표정을 짓고싶으면 입꼬리를 올리면 된다. 10초 동안 선택한 감정이 최대한 높게 나오게 표정을 다양하게 구사하면 된다.<br>
<div align="center">
<img width="500" alt="gameover" src="https://github.com/SoTaeHo/LOE-LeagueOfEmotion/assets/91146046/0a05768b-b208-4ddd-99e5-701310f6496d"><br>-결과창. 근소한 차이로 Player 2의 승리-  
</div>
게임이 종료되면 10초간 측정된 감정의 평균값을 구하여 최종 점수로 사용한다. 이후, 하단 Exit 버튼을 누르면 게임이 종료된다.


## 라이선스

MIT License

Copyright (c) [2024] [소태호]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## 참고 문헌

## 참고 문헌

- OpenCV 문서: [https://docs.opencv.org/](https://docs.opencv.org/)
- OpenCV DNN 모듈
  - [https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel](https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel),
  - [https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt) 
- NumPy 문서: [https://numpy.org/doc/](https://numpy.org/doc/)
- Matplotlib 문서: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- FER dataset: [https://www.kaggle.com/datasets/msambare/fer2013?select=train](https://www.kaggle.com/datasets/msambare/fer2013?select=train)
