
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization

# 경로 설정
train_dir = 'data/train'
test_dir = 'data/test'

# 이미지 데이터 제너레이터
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2,
    rotation_range=10,  # 낮은 회전 범위
    width_shift_range=0.1,  # 낮은 수평 이동 범위
    height_shift_range=0.1,  # 낮은 수직 이동 범위
    zoom_range=0.1,  # 낮은 줌 범위
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),  # MobileNetV2의 입력 크기
    color_mode='rgb',  # MobileNetV2는 RGB 이미지를 사용
    batch_size=64,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='rgb',
    batch_size=64,
    class_mode='categorical',
    subset='validation'
)

# 전이 학습을 위한 기본 모델 (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# 사전 학습된 모델 위에 새로운 층 추가
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(7, activation='softmax')(x)

# 전이 학습 모델 정의
model = Model(inputs=base_model.input, outputs=predictions)

# 사전 학습된 층을 동결
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10  # 적절한 epoch 수로 설정
)

# 모델 저장
model.save('/content/drive/MyDrive/optimized_emotion_model.h5')