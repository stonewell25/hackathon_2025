import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# 画像の読み込みと初期設定
overlay_image = cv2.imread('iwa-ku.jpg')  # ここに表示したい画像のパスを指定
if overlay_image is None:
    raise ValueError("画像が読み込めませんでした")

# 画像の初期サイズと位置
image_scale = 1.0
initial_width = 200  # 初期幅
initial_height = 200  # 初期高さ
image_x = 300  # 画像のx座標
image_y = 200  # 画像のy座標

# パンパース

# カメラからの映像をキャプチャ
cap = cv2.VideoCapture(1)

def calculate_finger_distance(hand_landmarks):
    # 親指と人差し指の先端の座標を取得
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # 距離を計算
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )
    return distance

def resize_and_overlay_image(frame, overlay, x, y, scale):
    # オーバーレイ画像のリサイズ
    new_width = int(initial_width * scale)
    new_height = int(initial_height * scale)
    resized_overlay = cv2.resize(overlay, (new_width, new_height))
    
    # 画像が画面内に収まるように位置を調整
    x = max(0, min(x, frame.shape[1] - new_width))
    y = max(0, min(y, frame.shape[0] - new_height))
    
    # アルファブレンディングのための準備
    overlay_height, overlay_width = resized_overlay.shape[:2]
    roi = frame[y:y+overlay_height, x:x+overlay_width]
    
    # 画像の合成
    result = cv2.addWeighted(roi, 0.5, resized_overlay, 0.5, 0)
    frame[y:y+overlay_height, x:x+overlay_width] = result
    
    return frame

# 前回の距離を保存する変数
prev_distance = None
base_scale = 1.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 画像を水平方向に反転
    frame = cv2.flip(frame, 1)

    # 画像をRGBに変換
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 手のランドマークの検出
    results = hands.process(image)

    # 画像をBGRに戻す
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 検出結果がある場合、ランドマークを描画し、ピンチジェスチャーを検出
    if results.multi_hand_landmarks:
        for hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks[1], mp_hands.HAND_CONNECTIONS)
            
            # 指の距離を計算
            current_distance = calculate_finger_distance(hand_landmarks[1])
            
            if prev_distance is not None:
                # 距離の変化に基づいてスケールを調整
                scale_change = (current_distance - prev_distance) * 5.0
                image_scale = max(0.5, min(2.0, base_scale + scale_change))
                base_scale = image_scale
            
            prev_distance = current_distance
    
    # 画像のオーバーレイ
    frame = resize_and_overlay_image(frame, overlay_image, image_x, image_y, image_scale)

    # 画像を表示
    cv2.imshow('Hand Gesture Image Control', frame)

    # キー入力をチェック
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# リソースの解放
cap.release()
cv2.destroyAllWindows()