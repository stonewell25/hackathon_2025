import cv2
import mediapipe as mp
import numpy as np
import math
import time

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# 画像の読み込み
overlay_image = cv2.imread('images/iwa-ku.jpg')
if overlay_image is None:
    raise ValueError("画像が読み込めませんでした")

# ビューポートのサイズ定義
VIEWPORT_WIDTH = 800
VIEWPORT_HEIGHT = 800
viewport_x = 100
viewport_y = 100

# 画像をビューポートに合わせて初期リサイズ
aspect_ratio = overlay_image.shape[1] / overlay_image.shape[0]
viewport_ratio = VIEWPORT_WIDTH / VIEWPORT_HEIGHT

if aspect_ratio > viewport_ratio:
    # 画像が横長の場合
    new_width = VIEWPORT_WIDTH
    new_height = int(new_width / aspect_ratio)
else:
    # 画像が縦長の場合
    new_height = VIEWPORT_HEIGHT
    new_width = int(new_height * aspect_ratio)

overlay_image = cv2.resize(overlay_image, (new_width, new_height))

# 状態管理用の定数
POINTING_THRESHOLD = 5  # ピクセル単位での移動閾値
POINTING_TIME_THRESHOLD = 0.1  # 秒単位での静止時間閾値

# 状態管理用の定数を更新
class PointingState:
    INITIAL_PLACEMENT = 0  # 初期配置モード
    NONE = 1          # 初期状態
    POINTING = 2      # 指差し中
    POINT_FIXED = 3   # ポイント確定
    ZOOMING = 4       # ズーム中

class ZoomDirection:
    NONE = 0
    ZOOM_IN = 1
    ZOOM_OUT = 2

# グローバル変数
current_state = PointingState.INITIAL_PLACEMENT
pointing_start_time = None
last_index_pos = None
fixed_point = None
current_scale = 1.0
base_scale = 1.0
prev_area = None

initial_position = None  # 画像の初期位置
placement_start_time = None  # 配置開始時間
PLACEMENT_TIME_THRESHOLD = 1.0  # 配置確定までの時間（秒）
last_placement_pos = None  # 最後の配置位置

# グローバル変数に追加
last_stable_scale = 1.0  # 最後に安定していたスケール
pinch_release_threshold = 0.15  # ピンチ解除を検出する閾値

SCALE_SMOOTHING_FACTOR = 0.3  # スケール変化の滑らかさ（0.1-0.5の間で調整）
MIN_SCALE_CHANGE = 0.01  # 最小スケール変化量
scale_velocity = 0.0  # スケール変化の速度

def check_finger_stable(current_pos, last_pos, threshold=5):
    """指の位置が安定しているかチェック"""
    if last_pos is None:
        return False
    distance = math.sqrt(
        (current_pos[0] - last_pos[0]) ** 2 +
        (current_pos[1] - last_pos[1]) ** 2
    )
    return distance < threshold

def get_index_finger_tip(hand_landmarks, frame_width, frame_height):
    """人差し指の先端の座標を取得"""
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    x = int(tip.x * frame_width)
    y = int(tip.y * frame_height)
    return (x, y)

def calculate_three_finger_area(hand_landmarks):
    """3本指（親指、人差し指、中指）の面積を計算"""
    finger_tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ]
    
    # 3点の座標を取得
    points = np.array([[tip.x, tip.y] for tip in finger_tips])
    
    # 3点で形成される三角形の面積を計算
    area = 0.5 * abs(np.cross(points[1] - points[0], points[2] - points[0]))
    return area

def is_three_finger_pinch(hand_landmarks):
    """3本指のピンチを検出"""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # 3本の指の距離を計算
    distances = [
        math.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2),
        math.sqrt((thumb_tip.x - middle_tip.x) ** 2 + (thumb_tip.y - middle_tip.y) ** 2),
        math.sqrt((index_tip.x - middle_tip.x) ** 2 + (index_tip.y - middle_tip.y) ** 2)
    ]
    
    # 全ての指が近い場合にTrue
    return all(d < 0.1 for d in distances)

def is_point_in_viewport(point):
    """指定された座標がビューポート内にあるかチェック"""
    x, y = point
    return (viewport_x <= x <= viewport_x + VIEWPORT_WIDTH and
            viewport_y <= y <= viewport_y + VIEWPORT_HEIGHT)

def calculate_scale_change(current_area, prev_area, hand_type):
    """手の種類に応じたスケール変化を計算"""

    """手の種類に応じたスケール変化を計算（スムージング機能付き）"""
    global scale_velocity

    if prev_area is None:
        return 0
    
    area_ratio = current_area / prev_area
    raw_scale_change = (area_ratio - 1.0)
    
    if hand_type == "Left":
        # 左手は拡大のみ
        if area_ratio > 1.0:  # エリアが増加している場合のみ
            target_scale_change = raw_scale_change * 1.5  # 感度調整
        else:
            target_scale_change = 0
    else:
        # 右手は縮小のみ
        if area_ratio < 1.0:  # エリアが減少している場合のみ
            target_scale_change = raw_scale_change * 1.0  # 感度調整
        else:
            target_scale_change = 0

    # スケール変化の速度を更新（スムージング）
    scale_velocity += (target_scale_change - scale_velocity) * SCALE_SMOOTHING_FACTOR
    
    # 微小な変化を無視
    if abs(scale_velocity) < MIN_SCALE_CHANGE:
        scale_velocity = 0
        return 0
        
    return scale_velocity

def apply_zoom(image, scale, center_x, center_y):
    """指定した中心点で画像をズーム"""
    if scale == 1.0:
        # スケールが1.0の場合は元のサイズのまま中央に配置
        # スケールが1.0の場合は元のサイズのまま中央に配置
        result = np.zeros((VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), dtype=np.uint8)
        y_start = (VIEWPORT_HEIGHT - image.shape[0]) // 2
        x_start = (VIEWPORT_WIDTH - image.shape[1]) // 2
        result[y_start:y_start+image.shape[0], 
               x_start:x_start+image.shape[1]] = image
        return result, (x_start, y_start)  # オフセットも返す
    
    
    # スケーリング後のサイズ
    scaled_width = int(image.shape[1] * scale)
    scaled_height = int(image.shape[0] * scale)
    
    # スケーリング
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))
    
    # 中心点に基づいてクロップ位置を計算
    crop_x = int(max(0, min(scaled_width - VIEWPORT_WIDTH,
                           center_x * scale - VIEWPORT_WIDTH / 2)))
    crop_y = int(max(0, min(scaled_height - VIEWPORT_HEIGHT,
                           center_y * scale - VIEWPORT_HEIGHT / 2)))
    
    # ビューポートサイズでクロップ
    try:
        cropped = scaled_image[crop_y:crop_y + VIEWPORT_HEIGHT,
                             crop_x:crop_x + VIEWPORT_WIDTH]
        
        if cropped.shape[:2] != (VIEWPORT_HEIGHT, VIEWPORT_WIDTH):
            result = np.zeros((VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), dtype=np.uint8)
            h, w = cropped.shape[:2]
            y_start = (VIEWPORT_HEIGHT - h) // 2
            x_start = (VIEWPORT_WIDTH - w) // 2
            result[y_start:y_start+h, x_start:x_start+w] = cropped
            return result, (x_start - crop_x, y_start - crop_y)  # オフセットを返す
        
        return cropped, (-crop_x, -crop_y)  # オフセットを返す
    except Exception as e:
        print(f"クロップエラー: {e}")
        return np.zeros((VIEWPORT_HEIGHT, VIEWPORT_WIDTH, 3), dtype=np.uint8), (0, 0)




def viewport_to_image_coords(point):
    """ビューポート座標系から画像座標系に変換"""
    x = point[0] - viewport_x
    y = point[1] - viewport_y
    return (x, y)


def is_open_palm_pose(hand_landmarks):
    """パーポーズ（全ての指が開いている状態）を検出"""
    # 各指のMCP関節（第一関節）とTIP（指先）の位置を取得
    finger_states = []
    
    # 人差し指
    index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # 中指
    middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # 薬指
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    
    # 小指
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # 親指は別途チェック（親指の付け根と先端）
    thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    
    # 各指が伸びているかチェック
    fingers = [
        (index_mcp, index_tip),
        (middle_mcp, middle_tip),
        (ring_mcp, ring_tip),
        (pinky_mcp, pinky_tip)
    ]
    
    # 通常の指について、指先がMCPよりも上にあるかチェック
    for mcp, tip in fingers:
        is_extended = tip.y < mcp.y  # 画像座標系では上が小さい値
        finger_states.append(is_extended)
    
    # 親指は横方向の開きをチェック
    thumb_extended = abs(thumb_tip.x - thumb_cmc.x) > 0.1
    finger_states.append(thumb_extended)
    
    # 全ての指が開いている場合にTrue
    return all(finger_states)

def reset_state():
    """状態をリセット"""
    global current_state, pointing_start_time, last_index_pos, fixed_point
    global current_scale, base_scale, prev_area, last_stable_scale, scale_velocity
    
    current_state = PointingState.INITIAL_PLACEMENT 
    pointing_start_time = None
    last_index_pos = None
    fixed_point = None
    current_scale = 1.0
    base_scale = 1.0
    prev_area = None
    last_stable_scale = 1.0
    scale_velocity = 0.0

# カメラキャプチャの設定
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # 画像の表示処理
    try:
        if current_state == PointingState.INITIAL_PLACEMENT:
            # 初期配置モード時の処理
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                index_pos = get_index_finger_tip(hand_landmarks, frame.shape[1], frame.shape[0])
                
                # 画像を指に追従させて表示
                viewport_x = index_pos[0] - VIEWPORT_WIDTH // 2
                viewport_y = index_pos[1] - VIEWPORT_HEIGHT // 2
                
                # 画面外にはみ出さないように調整
                viewport_x = max(0, min(viewport_x, frame.shape[1] - VIEWPORT_WIDTH))
                viewport_y = max(0, min(viewport_y, frame.shape[0] - VIEWPORT_HEIGHT))
                
                # 指の位置が安定しているかチェック
                if check_finger_stable(index_pos, last_placement_pos):
                    if placement_start_time is None:
                        placement_start_time = time.time()
                        cv2.putText(frame, 
                                  "Placing...", # 英語で表示
                                  (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  1,
                                  (0, 255, 0),
                                  2,
                                  cv2.LINE_AA)
                    elif time.time() - placement_start_time > PLACEMENT_TIME_THRESHOLD:
                        # 位置を確定
                        initial_position = (viewport_x, viewport_y)
                        current_state = PointingState.NONE
                        cv2.putText(frame, 
                                  "Placed!!", # 英語で表示
                                  (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  1,
                                  (0, 255, 0),
                                  2,
                                  cv2.LINE_AA)
                else:
                    placement_start_time = None
                
                last_placement_pos = index_pos
                
                # 仮の位置に画像を表示
                zoomed_image, _ = apply_zoom(overlay_image, 1.0, VIEWPORT_WIDTH // 2, VIEWPORT_HEIGHT // 2)
                frame[viewport_y:viewport_y+VIEWPORT_HEIGHT,
                      viewport_x:viewport_x+VIEWPORT_WIDTH] = zoomed_image
                
                # 配置中であることを示す枠を表示
                color = (0, 255, 0) if placement_start_time is not None else (0, 0, 255)
                cv2.rectangle(frame, 
                            (viewport_x, viewport_y),
                            (viewport_x + VIEWPORT_WIDTH, viewport_y + VIEWPORT_HEIGHT),
                            color, 2)
        
        else:
            # 通常モードでの表示処理（既存のコード）
            if initial_position is not None:
                viewport_x, viewport_y = initial_position
                if fixed_point is not None:
                    zoomed_image, offset = apply_zoom(
                        overlay_image,
                        current_scale,
                        fixed_point[0] - viewport_x,
                        fixed_point[1] - viewport_y
                    )
                else:
                    zoomed_image, offset = apply_zoom(
                        overlay_image,
                        1.0,
                        VIEWPORT_WIDTH // 2,
                        VIEWPORT_HEIGHT // 2
                    )
                
                frame[viewport_y:viewport_y+VIEWPORT_HEIGHT,
                      viewport_x:viewport_x+VIEWPORT_WIDTH] = zoomed_image
                
                cv2.rectangle(frame, 
                            (viewport_x, viewport_y),
                            (viewport_x + VIEWPORT_WIDTH, viewport_y + VIEWPORT_HEIGHT),
                            (0, 255, 0), 2)
    
    except Exception as e:
        print(f"表示エラー: {e}")

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # 手の種類を判定（右手/左手）
            hand_type = handedness.classification[0].label
            confidence = handedness.classification[0].score
            # 手の位置を取得
            index_pos = get_index_finger_tip(hand_landmarks, frame.shape[1], frame.shape[0])
            
            # グーポーズのチェック
            if is_point_in_viewport(index_pos):
                if is_open_palm_pose(hand_landmarks):
                    if current_state != PointingState.NONE:  # 現在の状態が初期状態でない場合のみリセット
                        reset_state()
                        # リセット時のフィードバック表示
                        cv2.putText(frame, 
                                "Reset!!", # 英語で表示
                                (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA)
                        continue  # 他の処理をスキップ
            
        # 状態に応じた処理
        if current_state == PointingState.NONE:
            # 人差し指の位置を記録開始
            if last_index_pos is None:
                last_index_pos = index_pos
                pointing_start_time = time.time()
                current_state = PointingState.POINTING
        
        elif current_state == PointingState.POINTING:
            # ビューポート外に出た場合はポインティングをキャンセル
            if not is_point_in_viewport(index_pos):
                current_state = PointingState.NONE
                last_index_pos = None
                pointing_start_time = None
            else:
                # 指の移動量をチェック
                movement = math.sqrt(
                    (index_pos[0] - last_index_pos[0]) ** 2 +
                    (index_pos[1] - last_index_pos[1]) ** 2
                )
                if movement < POINTING_THRESHOLD:
                    # 指が十分に静止している場合
                    if time.time() - pointing_start_time > POINTING_TIME_THRESHOLD:
                        # ポイント確定
                        fixed_point = index_pos
                        current_state = PointingState.POINT_FIXED
                        print("ポイント確定")
                else:
                    # 移動が大きい場合は時間をリセット
                    pointing_start_time = time.time()
                    last_index_pos = index_pos
        
        elif current_state == PointingState.POINT_FIXED:
            if is_point_in_viewport(index_pos):
                # 3本指のピンチを検出
                if is_three_finger_pinch(hand_landmarks):
                    current_state = PointingState.ZOOMING
                    current_area = calculate_three_finger_area(hand_landmarks)
                    prev_area = current_area
                    base_scale = current_scale
                   

        # メインループ内のズーム処理部分も修正
        elif current_state == PointingState.ZOOMING:
            if is_point_in_viewport(index_pos):
                if is_three_finger_pinch(hand_landmarks):
                    current_area = calculate_three_finger_area(hand_landmarks)
                    if prev_area is not None:
                        scale_change = calculate_scale_change(current_area, prev_area, hand_type)
                        if scale_change != 0:  # スケール変化がある場合のみ更新
                            new_scale = base_scale * (1.0 + scale_change)
                            current_scale = max(1.0, min(5.0, new_scale))
                            last_stable_scale = current_scale
                    prev_area = current_area
                else:
                    current_scale = last_stable_scale
                    current_state = PointingState.POINT_FIXED
                    prev_area = None
                    
        # 状態に応じた描画
        if fixed_point is not None:
            # 確定したポイントを表示
            cv2.circle(frame, fixed_point, 5, (0, 255, 0), -1)
        
        if current_state == PointingState.POINTING:
            # 指差し中の点を表示
            # 指差し中の点を表示（ビューポート内の場合は青、外の場合は赤）
            color = (255, 0, 0) if is_point_in_viewport(index_pos) else (0, 0, 255)
            cv2.circle(frame, index_pos, 5, color, -1)
        
        # ランドマークの描画
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 画像の表示処理
    try:
        if fixed_point is not None:
            zoomed_image, offset = apply_zoom(
                overlay_image,
                current_scale,
                fixed_point[0] - viewport_x,
                fixed_point[1] - viewport_y
            )
        else:
            zoomed_image, offset = apply_zoom(
                overlay_image,
                1.0,
                VIEWPORT_WIDTH // 2,
                VIEWPORT_HEIGHT // 2
            )
            
        # 画像をフレームに合成
        frame[viewport_y:viewport_y+VIEWPORT_HEIGHT,
              viewport_x:viewport_x+VIEWPORT_WIDTH] = zoomed_image
        
               # 指さしポイントとビューポートの枠を描画
        # 指さしポイントとビューポートの枠を描画
        if results.multi_hand_landmarks:
            index_pos = get_index_finger_tip(hand_landmarks, frame.shape[1], frame.shape[0])
            
            if is_point_in_viewport(index_pos):
                # ビューポート内の座標に変換
                image_point = viewport_to_image_coords(index_pos)
                
                if current_state == PointingState.POINTING:
                    # スケールとオフセットを考慮した座標に描画
                    adjusted_point = (
                        int(image_point[0] + offset[0]),
                        int(image_point[1] + offset[1])
                    )
                    cv2.circle(zoomed_image, adjusted_point, 5, (255, 0, 0), -1)
                    cv2.circle(zoomed_image, adjusted_point, 15, (255, 0, 0), 2)
                
                if fixed_point is not None and current_state in [PointingState.POINT_FIXED, PointingState.ZOOMING]:
                    # 確定したポイントをスケールとオフセットを考慮して描画
                    fixed_point_image = viewport_to_image_coords(fixed_point)
                    adjusted_fixed_point = (
                        int(fixed_point_image[0] + offset[0]),
                        int(fixed_point_image[1] + offset[1])
                    )
                    
                    # 十字マークを描画
                    size = 10
                    cv2.circle(zoomed_image, adjusted_fixed_point, 5, (0, 255, 0), -1)
                    cv2.line(zoomed_image, 
                            (adjusted_fixed_point[0] - size, adjusted_fixed_point[1]),
                            (adjusted_fixed_point[0] + size, adjusted_fixed_point[1]),
                            (0, 255, 0), 2)
                    cv2.line(zoomed_image, 
                            (adjusted_fixed_point[0], adjusted_fixed_point[1] - size),
                            (adjusted_fixed_point[0], adjusted_fixed_point[1] + size),
                            (0, 255, 0), 2)

        # 更新された画像をフレームに再配置
        frame[viewport_y:viewport_y+VIEWPORT_HEIGHT,
            viewport_x:viewport_x+VIEWPORT_WIDTH] = zoomed_image
        
        # ビューポートの枠を描画
        cv2.rectangle(frame, 
                    (viewport_x, viewport_y),
                    (viewport_x + VIEWPORT_WIDTH, viewport_y + VIEWPORT_HEIGHT),
                    (0, 255, 0), 2)

    except Exception as e:
        print(f"表示エラー: {e}")
    # 状態表示
    status_text = {
        PointingState.INITIAL_PLACEMENT: "initial placement",
        PointingState.NONE: "waiting",
        PointingState.POINTING: "pointing",
        PointingState.POINT_FIXED: "fixed",
        PointingState.ZOOMING: "zooming"
    }[current_state]
    
    cv2.putText(frame, 
                f"{status_text} (Scale: {current_scale:.2f})",
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA)

    cv2.imshow('Image Viewer', frame)

    # キー入力処理
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # 状態をリセット
        current_state = PointingState.NONE
        pointing_start_time = None
        last_index_pos = None
        fixed_point = None
        current_scale = 1.0
        base_scale = 1.0
        prev_area = None

cap.release()
cv2.destroyAllWindows()