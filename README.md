# LOE - League Of Emotion

LOE - League Of Emotion은 FER(Face Emotion Recognition)을 이용하여 누가 더 감정을 생생하게 표현하는지 겨뤄보는 게임이다.<br/>
감정은 총 7가지(angry, disgust, fear, happy, neutral, sad, surprise)가 있으며, 게임 시작 시 어떤 감정으로 겨룰지 선택할 수 있다.<br/><br/>
감정을 선택하면 Player 1부터 10초동안 웹캠을 바라보고 선택한 감정이 잘 드러나게 표정을 지으면 된다. 10초동안 Player 1의 얼굴을 감지하여 해당 얼굴이 나타내는 감정의 점수를 실시간으로 보여주고 기록한다. Player 1의 차례가 끝나면 Player 2의 차례가 시작되며, 동일하게 10초동안 웹캠을 향해 표정을 지으면 감정의 점수를 측정한다. Player 2의 측정이 끝나면, 10초 동안의 감정의 평균을 구하여 결과창에서 누가 승리하였는지 보여준다.

## 목차
- [특징](#특징)
- [요구 사항](#요구-사항)
- [모델 학습](#모델-학습)
- [화면 출력](#화면-출력)
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

얼굴 감지기는 기존에는 Haar Cascade 감지기를 사용했었다. 얼굴 감지할 때 빠른 속도가 장점이라하여 사용했는데, 점수 측정이나 실시간 연산이 계속되다보니 빠른 속도가 체감되지 않았다. 또한 어두운 환경이거나 얼굴의 측면을 보이면 제대로 감지하지 못했다. 따라서 더 좋은 성능을 보여주는 OpenCV DNN 모듈을 사용하게 되었다. 속도는 별 차이 없었지만, 얼굴의 측면을 훨씬 잘 감지하여 프로그램의 성능을 향상 시킬 수 있었다.

### ImageDataGenerator

- `ImageDataGenerator`를 사용하여 이미지 데이터를 전처리한다. 데이터는 0과 1 사이의 값으로 스케일링되고, train data의 20%는 validation data로 사용한다.

### 데이터 생성기

- `flow_from_directory`를 사용하여 train data, validation data, test data를 생성한다. 각 이미지의 크기는 48x48로 조정되며, 회색조로 변환된다. 배치 크기는 64로 설정되고, 다중 클래스 분류를 위해 `categorical` 모드를 사용했다.

### 모델 구축

- `Sequential` 모델을 사용하여 3개의 합성곱 층(Conv2D)과 맥스 풀링(MaxPooling2D) 및 드롭아웃(Dropout) 층으로 구성된 신경망을 구축했다. 마지막에 두 개의 전결합 층(Dense)이 있으며, 7개의 감정 클래스를 분류하기 위해 소프트맥스 활성화 함수를 사용했다.

### 모델 컴파일

- `adam` 옵티마이저와 `categorical_crossentropy`를 사용하여 모델을 컴파일한다. 성능 평가는 정확도를 측정했다.

### 모델 학습

- 모델은 train data를 사용하여 30 epoch 학습했다. 학습 중 validation data가 사용되어 모델의 성능을 평가한다.

## 화면 출력

openCV의 기능 내에서 해결하였다. 

### select_emotion, exit_program

- 마우스 이벤트를 다루는 함수이다. 감정을 고를 때, exit버튼을 눌러 종료할 때, 호출된다.

### show_emotion_selection, countdown

- 감정 선택 창과, 카운트 다운 화면을 불러와주는 함수이다.

### load_face_detector

- 얼굴 감지기 모델을 불러오는 함수이다.

### detect_faces_dnn

- load한 얼굴 감지기를 이용하여 프레임에서 얼굴을 감지한다.

### analyze_emotion

- 10초동안 프레임을 캡처하여 감정을 분석한다.
- 얼굴을 감지하고, 감지된 얼굴 영역을 모델의 입력으로 넘겨 감정을 예측하여 감정 점수를 나타내준다.

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
  <img width="400" alt="player1" src="https://github.com/SoTaeHo/LOE-LeagueOfEmotion/assets/91146046/5be27ebc-9448-4c6c-8e8c-f7f139ecdb9b" style="display:inline-block; margin-right: 10px;">
  <img width="400" alt="player2" src="https://github.com/SoTaeHo/LOE-LeagueOfEmotion/assets/91146046/2201b412-3c70-414c-93cd-56c29ffb85d0" style="display:inline-block;"><br>
  -우는 표정을 지으면 sad의 점수가 올라가고, 웃는 표정을 지으면 happy의 점수가 올라간다-
</div>

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

- OpenCV 문서: [https://docs.opencv.org/](https://docs.opencv.org/)
- OpenCV DNN 모듈
  - [https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel](https://github.com/gopinath-balu/computer_vision/blob/master/CAFFE_DNN/res10_300x300_ssd_iter_140000.caffemodel),
  - [https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt) 
- NumPy 문서: [https://numpy.org/doc/](https://numpy.org/doc/)
- Matplotlib 문서: [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
- FER dataset: [https://www.kaggle.com/datasets/msambare/fer2013?select=train](https://www.kaggle.com/datasets/msambare/fer2013?select=train)
