import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import time

# 감정 인식 모델 로드
model = load_model('emotion_model.h5')

# 감정 라벨
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 감정 선택 변수 초기화
selected_emotion = None

# 마우스 클릭 이벤트 핸들러
def select_emotion(event, x, y, flags, param):
    global selected_emotion
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, emotion in enumerate(emotion_labels):
            if 10 <= x <= 200 and 30 + i * 30 <= y <= 60 + i * 30:
                selected_emotion = emotion
                break

def exit_program(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 150 and 450 <= y <= 480:
            cv2.destroyAllWindows()
            exit()

# 감정 선택 창 표시
def show_emotion_selection(frame):
    global selected_emotion
    while selected_emotion is None:
        frame[:] = (50, 50, 50)
        for i, emotion in enumerate(emotion_labels):
            color = (0, 255, 0) if selected_emotion == emotion else (255, 255, 255)
            cv2.putText(frame, emotion, (10, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow('Emotion Recognition', frame)
        cv2.setMouseCallback('Emotion Recognition', select_emotion)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 카운트다운 표시
def countdown(cap, frame, duration=3):
    for i in range(duration, 0, -1):
        start_time = time.time()
        while time.time() - start_time < 1:
            ret, cap_frame = cap.read()
            if not ret:
                break
            cap_frame = cv2.resize(cap_frame, (640, 480))
            cv2.putText(cap_frame, str(i), (270, 240), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
            cv2.imshow('Emotion Recognition', cap_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    start_time = time.time()
    while time.time() - start_time < 1:
        ret, cap_frame = cap.read()
        if not ret:
            break
        cap_frame = cv2.resize(cap_frame, (640, 480))
        cv2.putText(cap_frame, "Start!", (170, 240), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
        cv2.imshow('Emotion Recognition', cap_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 얼굴 감지기 로드
def load_face_detector():
    prototxt_path = 'deploy.prototxt'
    caffemodel_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    return net

# 얼굴 감지
def detect_faces_dnn(net, frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    faces = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))

    return faces

# 감정 분석 수행
def analyze_emotion(frame, cap, net, player, duration=10):
    frame_count = 0
    emotion_scores = []

    start_time = time.time()
    while time.time() - start_time < duration:
        ret, cap_frame = cap.read()
        if not ret:
            break

        cap_frame = cv2.resize(cap_frame, (640, 480))
        gray = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2GRAY)

        faces = detect_faces_dnn(net, cap_frame)
        emotion_probs = None
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray / 255.0
            roi_gray = roi_gray.reshape(1, 48, 48, 1)
            emotion_probs = model.predict(roi_gray)[0]
            emotion_scores.append(emotion_probs)
            cv2.rectangle(cap_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # 플레이어 정보 표시
        cv2.putText(cap_frame, f'Player {player}\'s Turn', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 감정 점수 표시
        if emotion_probs is not None:
            h, w, _ = cap_frame.shape
            for i, (emotion, score) in enumerate(zip(emotion_labels, emotion_probs)):
                text = f"{emotion}: {score:.6f}"
                cv2.putText(cap_frame, text, (w - 250, h - (7 - i) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 현재 화면 갱신
        frame[:cap_frame.shape[0], :cap_frame.shape[1]] = cap_frame
        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return np.mean(emotion_scores, axis=0)

# 메인 함수
def main():
    global selected_emotion
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 감정 선택
    show_emotion_selection(frame)

    # 웹캠 연결
    cap = cv2.VideoCapture(0)

    # 얼굴 감지기 로드
    net = load_face_detector()

    # 첫 번째 플레이어 감정 분석
    countdown(cap, frame)
    score1 = analyze_emotion(frame, cap, net, player=1)
    print(f"Player 1's {selected_emotion} score: {score1}")

    # 두 번째 플레이어 감정 분석
    countdown(cap, frame)
    score2 = analyze_emotion(frame, cap, net, player=2)
    print(f"Player 2's {selected_emotion} score: {score2}")

    # 결과 표시
    selected_emotion_index = emotion_labels.index(selected_emotion)
    score1_selected_emotion = score1[selected_emotion_index]
    score2_selected_emotion = score2[selected_emotion_index]

    if score1_selected_emotion > score2_selected_emotion:
        winner = "Player 1"
    elif score1_selected_emotion < score2_selected_emotion:
        winner = "Player 2"
    else:
        winner = "It's a tie!"

    print(f"The winner is: {winner}")
    frame[:] = (0, 0, 0)
    cv2.putText(frame, f"Player 1's {selected_emotion} score: {score1_selected_emotion:.6f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Player 2's {selected_emotion} score: {score2_selected_emotion:.6f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"The winner is: {winner}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Exit", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Emotion Recognition', frame)
    cv2.setMouseCallback('Emotion Recognition', exit_program)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
