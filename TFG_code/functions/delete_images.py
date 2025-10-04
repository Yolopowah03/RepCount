import os

for subdir, _, files in os.walk('/datatmp2/joan/tfg_joan/LSTM_dataset/train/labels/squat'):
    for file in files:
        if not file.endswith('.json'):
            os.remove(os.path.join('/datatmp2/joan/tfg_joan/LSTM_dataset/train/labels/squat', subdir, file))