import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd

ROOT_DATA_DIR = 'IPN_Hand' 
ANNOTATION_FILE = 'IPN_Hand/annotations/annotations/Annot_List.txt'
OUTPUT_DIR = 'MP_Data'
SEQUENCE_LENGTH = 30

LABEL_MAP = {
    1: 'Idle', 2: 'Point_1F', 3: 'Point_2F', 4: 'Click_1F', 5: 'Click_2F',
    6: 'Swipe_Up', 7: 'Swipe_Down', 8: 'Swipe_Left', 9: 'Swipe_Right',
    10: 'Open_Twice', 11: 'DbClick_1F', 12: 'DbClick_2F', 13: 'Zoom_In', 14: 'Zoom_Out'
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        res = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.landmark])
        wrist = landmarks[0]
        return (landmarks - wrist).flatten()
    return np.zeros(21*3)

def build_video_map(root_dir):
    print(f"Đang quét toàn bộ thư mục '{root_dir}' để tìm video...")
    video_map = {}
    count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.avi', '.mp4', '.mov')):
                full_path = os.path.join(root, file)
                name_no_ext = os.path.splitext(file)[0]
                video_map[name_no_ext] = full_path
                count += 1
    
    print(f"Đã lập bản đồ vị trí của {count} video!")
    return video_map

def process_ipn_dataset():
    video_paths = build_video_map(ROOT_DATA_DIR)
    
    if len(video_paths) == 0:
        print("LỖI: Không tìm thấy bất kỳ video nào! Kiểm tra lại folder IPN_Hand.")
        return

    print(f"Đang đọc file: {ANNOTATION_FILE}")
    try:
        df = pd.read_csv(ANNOTATION_FILE)
    except Exception as e:
        print(f"Lỗi đọc CSV: {e}")
        return

    print(f"Tìm thấy {len(df)} dòng dữ liệu trong file text.")
    
    action_counts = {name: 0 for name in LABEL_MAP.values()}
    processed_count = 0

    for index, row in df.iterrows():
        try:
            video_name = str(row['video']).strip()
            label_id = int(row['id'])
            start_frame = int(row['t_start'])
            end_frame = int(row['t_end'])
        except:
            continue
            
        if label_id not in LABEL_MAP: continue 

        if video_name in video_paths:
            found_video_path = video_paths[video_name]
        else:
            # Chỉ báo lỗi 5 lần đầu
            if processed_count < 5:
                print(f"Không khớp: File text ghi '{video_name}' nhưng không tìm thấy file video tương ứng.")
            continue

        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Đã xử lý {processed_count} mẫu... (Video: {video_name})")

        action_name = LABEL_MAP[label_id]
        cap = cv2.VideoCapture(found_video_path)
        frames_data = []
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret: break
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            frames_data.append(extract_keypoints(results))
        cap.release()

        # Lưu file
        if len(frames_data) > 5: 
            if len(frames_data) > SEQUENCE_LENGTH:
                start = (len(frames_data) - SEQUENCE_LENGTH) // 2
                final_sequence = frames_data[start : start + SEQUENCE_LENGTH]
            else:
                final_sequence = frames_data[:]
                while len(final_sequence) < SEQUENCE_LENGTH:
                    final_sequence.append(np.zeros(21*3))
            
            save_path = os.path.join(OUTPUT_DIR, action_name, str(action_counts[action_name]))
            os.makedirs(save_path, exist_ok=True)
            for i, kp in enumerate(final_sequence):
                np.save(os.path.join(save_path, f"{i}.npy"), kp)
            
            action_counts[action_name] += 1

    print(f"HOÀN TẤT! Tổng cộng đã lưu: {sum(action_counts.values())} mẫu.")

if __name__ == "__main__":
    process_ipn_dataset()