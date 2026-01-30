import numpy as np
import os
import json
import joblib
import zipfile
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# File này được thiết kế để chạy trên GOOGLE COLAB.
# Điều chỉnh đường dẫn nếu chạy trên máy cá nhân.
ZIP_PATH = "/content/drive/MyDrive/Project_Hand/MP_Data.zip"
DATA_PATH = "/content/MP_Data"
SAVE_DIR = '/content/drive/MyDrive/Project_Hand/Model_Output'
SEQUENCE_LENGTH = 30
INPUT_DIM = 63

ACTION_MAPPING = {
    'Point_1F':   'Move',
    'Point_2F':   'Move',
    'Click_1F':   'LeftClick',
    'DbClick_1F': 'LeftClick',
    'Click_2F':   'RightClick',
    'DbClick_2F': 'RightClick',
    'Swipe_Left':  'NextSlide',
    'Swipe_Right': 'PrevSlide',
    'Swipe_Up':    'ScrollUp',
    'Swipe_Down':  'ScrollDown',
    'Zoom_In':     'ZoomIn',
    'Zoom_Out':    'ZoomOut',
    'Idle':        'Idle'
}


if not os.path.exists(DATA_PATH):
    print(f"Chua thay du lieu. Dang giai nen zip: {ZIP_PATH}")
    if os.path.exists(ZIP_PATH):
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall("/content/")
        print("Giai nen XONG!")
    else:
        print("LOI: Khong tim thay file Zip!")
        exit()


actions = np.array(sorted(list(set(ACTION_MAPPING.values()))))
label_map = {label: num for num, label in enumerate(actions)}

def augment_data(sequence):
    noise = np.random.normal(0, 0.015, sequence.shape)
    return sequence + noise

sequences, labels = [], []
print(f"\nDang quet du lieu tai: {DATA_PATH}")

for original_action in os.listdir(DATA_PATH):
    if original_action not in ACTION_MAPPING: continue

    target_label = ACTION_MAPPING[original_action]
    action_path = os.path.join(DATA_PATH, original_action)

    if not os.path.isdir(action_path): continue


    files = [f for f in os.listdir(action_path) if f.endswith('.npy')]


    sub_folders = [f for f in os.listdir(action_path) if os.path.isdir(os.path.join(action_path, f))]

    if len(files) > 0:

        print(f"-> {original_action}: Phat hien {len(files)} files (NEW FORMAT)")
        for file_name in files:
            try:
                window = np.load(os.path.join(action_path, file_name))
                if window.shape == (SEQUENCE_LENGTH, INPUT_DIM):
                    sequences.append(window)
                    labels.append(label_map[target_label])
                    sequences.append(augment_data(window))
                    labels.append(label_map[target_label])
            except: pass

    elif len(sub_folders) > 0:

        print(f"-> {original_action}: Phat hien {len(sub_folders)} folders (OLD FORMAT)")
        for sample_folder in sub_folders:
            sample_path = os.path.join(action_path, sample_folder)
            window = []

            for frame_num in range(SEQUENCE_LENGTH):
                res_path = os.path.join(sample_path, f"{frame_num}.npy")
                if os.path.exists(res_path):
                    res = np.load(res_path)
                    window.append(res)

            if len(window) == SEQUENCE_LENGTH:
                window = np.array(window)
                sequences.append(window)
                labels.append(label_map[target_label])
                sequences.append(augment_data(window))
                labels.append(label_map[target_label])
    else:
        print(f"-> {original_action}: Khong tim thay du lieu hop le (Empty)")

print(f"\nTong cong tim thay: {len(sequences)} mau du lieu.")

if len(sequences) == 0:
    print("LOI: Van khong tim thay du lieu. Kiem tra lai folder MP_Data.")
    exit()


X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions)).astype(int)


try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)
except ValueError:
    print("Canh bao: Mot so class qua it du lieu de stratify. Chuyen sang che do split thuong.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

scaler = StandardScaler()
N_train, L, D = X_train.shape
N_test, _, _ = X_test.shape

X_train_reshaped = X_train.reshape(-1, D)
X_test_reshaped = X_test.reshape(-1, D)
scaler.fit(X_train_reshaped)
X_train = scaler.transform(X_train_reshaped).reshape(N_train, L, D)
X_test = scaler.transform(X_test_reshaped).reshape(N_test, L, D)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(SEQUENCE_LENGTH, INPUT_DIM), kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.4))
model.add(LSTM(32, return_sequences=False, activation='tanh', kernel_regularizer=regularizers.l2(0.001)))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(BatchNormalization())
model.add(Dense(len(actions), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    shuffle=True,
    callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)]
)

model.save(os.path.join(SAVE_DIR, 'action_best_model.h5'))
joblib.dump(scaler, os.path.join(SAVE_DIR, 'scaler.save'))
with open(os.path.join(SAVE_DIR, 'actions.json'), 'w') as f:
    json.dump(actions.tolist(), f)

print(f"\nTHANH CONG! Model da luu tai: {SAVE_DIR}")