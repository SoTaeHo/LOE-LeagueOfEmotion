# LOE - League Of Emotion

LOE - League Of Emotion은 FER(Face Emotion Recognition)을 이용하여 누가 더 감정을 생생하게 표현하는지 겨뤄보는 게임입니다.<br/>
감정은 총 7가지(angry, disgust, fear, happy, neutral, sad, surprise)가 있으며, 게임 시작 시 어떤 감정으로 겨룰지 선택할 수 있습니다.<br/><br/>
감정을 선택하면 Player 1부터 10초동안 웹캠을 바라보고 선택한 감정이 잘 드러나게 표정을 지으면 됩니다. 10초동안 Player 1의 얼굴을 감지하여 해당 얼굴이 나타내는 감정의 점수를 실시간으로 보여주고 기록합니다. Player 1의 차례가 끝나면 Player 2의 차례가 시작되며, 동일하게 10초동안 웹캠을 향해 표정을 지으면 감정의 점수를 측정합니다. Player 2의 측정이 끝나면, 10초 동안의 감정의 평균을 구하여 결과창에서 누가 승리하였는지 보여줍니다.

## 목차
- [특징](#특징)

## 특징

...

## 요구 사항

...

## 설치 방법

...

## 사용법

### 실시간 감정 감지

...

### 이미지 기반 감정 감지


## 기여

...

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
