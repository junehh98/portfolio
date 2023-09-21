import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2


# 학습 이미지, json파일
image_folder = '/Users/junehh98/Desktop/final/290.이안류 CCTV 데이터/01.데이터/Training/01.원천데이터/TS.이미지.해운대_2.PARA1'
json_folder = '/Users/junehh98/Desktop/final/290.이안류 CCTV 데이터/01.데이터/Training/02.라벨링데이터/TL.JSON.해운대_2.PARA1'
# 테스트 이미지
test_image_folder = '/Users/junehh98/Desktop/final/290.이안류 CCTV 데이터/01.데이터/Validation/01.원천데이터/VS.이미지.해운대_2.PARA1'

# 파일 이름이 같지 않아서 파일을 가져오는 순서대로 매칭
image_files = sorted(os.listdir(image_folder))
json_files = sorted(os.listdir(json_folder))

data_images = []  # 학습 이미지
data_labels = []  # 클래스 분류
data_coordinates = []  # 이안류가 일어난 부분

# 이미지파일과 json파일 묶어주기
for img_file, json_file in tqdm(zip(image_files, json_files), desc="Loading Training Data"):
    img_path = os.path.join(image_folder, img_file)
    json_path = os.path.join(json_folder, json_file)

    try:
        img = Image.open(img_path)  # 이미지 열기
        # 이미지 사이즈를 255x255로 조정
        img_resized = img.resize((255, 255))
        # 조정된 이미지 numpy 배열로 추가
        data_images.append(np.array(img_resized))

        with open(json_path, 'r') as f:
            data = json.load(f)
            # json파일의 필요한 부분만 사용
            # annotations의 class 라벨 -> 1이면 이안류 존재
            class_label = data['annotations']['class']
            data_labels.append(class_label)

            # json파일에 이미지의 어떤 부분이 이안류인지 알려주는 drawing라벨에 numpy값이 있음
            if 'drawing' in data['annotations']:
                coordinates = []  # 변환된 numpy값 저장

                for drawing in data['annotations']['drawing']:
                    # 좌표 조정, drawing라벨의 numpy값도 255x255로 줄여줘야함
                    coordinate = np.array(drawing) * [255 / img.width, 255 / img.height]
                    coordinates.append(coordinate)
                data_coordinates.append(coordinates)
            else:
                data_coordinates.append([])

    # 예외처리
    except Exception as e:
        print(f"Error processing image '{img_file}': {str(e)}")
        continue

# 이미지와 라벨을 numpy 변환
data_images = np.array(data_images)
data_labels = np.array(data_labels)

# 이미지를 텐서로 변환
data_images = tf.convert_to_tensor(data_images)

# 이미지의 크기 정보
image_height, image_width = data_images[0].shape[:2]

# 클래스 불균형 비율 계산
total_samples = len(data_labels)
positive_samples = np.sum(data_labels == 1)
negative_samples = np.sum(data_labels == 0)
positive_ratio = positive_samples / total_samples
negative_ratio = negative_samples / total_samples

# 클래스 가중치 계산
class_weights = {0: 1 / negative_ratio, 1: 1 / positive_ratio}

# 모델 생성
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(image_height, image_width, 3))
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# 조기종료 옵션 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 모델 학습 시 클래스 가중치 적용
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(data_images, data_labels, batch_size=32, epochs=1, validation_split=0.2, callbacks=[early_stopping], class_weight=class_weights)

# 테스트 이미지 처리

# Load test images
test_images = []
for test_img_file in sorted(os.listdir(test_image_folder)):
    test_img_path = os.path.join(test_image_folder, test_img_file)
    test_img = Image.open(test_img_path).resize((255, 255))
    test_images.append(np.array(test_img))

test_images = np.array(test_images)

# 테스트 이미지를 텐서로 변환
test_images = tf.convert_to_tensor(test_images)

# Preprocess test images
preprocessed_test_images = preprocess_input(test_images)

# Make predictions on test images
predictions = model.predict(preprocessed_test_images)

# Visualize the results
for i in range(len(test_images)):
    test_img = test_images[i]
    prediction = predictions[i]

    # Threshold the prediction
    threshold = 0.5
    predicted_class = 1 if prediction >= threshold else 0

    # Get the 이안류-affected region coordinates
    insect_coordinates = data_coordinates[i]  # Assuming data_coordinates corresponds to the test images

    # Draw bounding box on the image if 이안류 is detected
    if predicted_class == 1 and insect_coordinates:
        img = Image.fromarray(test_img)
        draw = ImageDraw.Draw(img)

        # Draw a rectangle for each 이안류-affected region
        for coordinate in insect_coordinates:
            # Ensure the coordinate shape is (8,)
            if len(coordinate) == 4:
                x1, y1, x2, y2 = coordinate
                # Convert normalized coordinates to pixel coordinates
                x1 *= img.width
                y1 *= img.height
                x2 *= img.width
                y2 *= img.height
                draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)

        img.show()
