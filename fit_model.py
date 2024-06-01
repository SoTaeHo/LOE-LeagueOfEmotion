# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization
# import keras_tuner as kt
#
# # 경로 설정
# train_dir = '/data/train'
# test_dir = '/data/test'
#
# # 이미지 데이터 제너레이터
# datagen = ImageDataGenerator(
#     rescale=1. / 255,
#     validation_split=0.2,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
#
# train_generator = datagen.flow_from_directory(
#     train_dir,
#     target_size=(48, 48),
#     color_mode='grayscale',
#     batch_size=64,
#     class_mode='categorical',
#     subset='training'
# )
#
# validation_generator = datagen.flow_from_directory(
#     train_dir,
#     target_size=(48, 48),
#     color_mode='grayscale',
#     batch_size=64,
#     class_mode='categorical',
#     subset='validation'
# )
#
# # 전이 학습을 위한 기본 모델 (VGG16)
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))
#
# # 사전 학습된 모델 위에 새로운 층 추가
# x = base_model.output
# x = Flatten()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(7, activation='softmax')(x)
#
# # 전이 학습 모델 정의
# transfer_model = Model(inputs=base_model.input, outputs=predictions)
#
# # 사전 학습된 층을 동결
# for layer in base_model.layers:
#     layer.trainable = False
#
# # 모델 컴파일
# transfer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 하이퍼파라미터 튜닝을 위한 모델 빌더 함수
# def build_model(hp):
#     model = Sequential()
#     model.add(Conv2D(
#         filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
#         kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]),
#         activation='relu',
#         input_shape=(48, 48, 1)
#     ))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(
#         filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
#         kernel_size=hp.Choice('conv_2_kernel', values=[3, 5]),
#         activation='relu'
#     ))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(
#         filters=hp.Int('conv_3_filter', min_value=32, max_value=128, step=16),
#         kernel_size=hp.Choice('conv_3_kernel', values=[3, 5]),
#         activation='relu'
#     ))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(
#         units=hp.Int('dense_units', min_value=64, max_value=256, step=32),
#         activation='relu'
#     ))
#     model.add(BatchNormalization())
#     model.add(Dropout(0.5))
#     model.add(Dense(7, activation='softmax'))
#
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model
#
# # 하이퍼파라미터 튜너 설정
# tuner = kt.Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=10,
#     hyperband_iterations=2
# )
#
# # 하이퍼파라미터 튜닝
# tuner.search(train_generator, epochs=30, validation_data=validation_generator)
#
# # 최적의 하이퍼파라미터를 사용한 모델
# best_model = tuner.get_best_models(num_models=1)[0]
#
# # 최적의 모델에 전이 학습의 가중치 적용
# best_model.layers[0].set_weights(transfer_model.layers[0].get_weights())
#
# # 전이 학습 모델의 컴파일 설정을 최적 모델에 적용
# best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # 모델 학습
# history = best_model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
#     epochs=30
# )
#
# # 모델 저장
# best_model.save('best_emotion_model.h5')


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization

# GPU 사용 설정 확인
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 경로 설정
train_dir = '../data/train'
test_dir = '../data/test'

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
model.save('optimized_emotion_model.h5')
