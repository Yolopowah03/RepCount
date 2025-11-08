# Requisitos: torch, numpy, sklearn
import os, math, random
import numpy as np
import torch #type: ignore
import torch.nn as nn #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix #type: ignore
import json
import shutil

# COCO_KPTS_INFO = [
  # 0: nose
  # 1: left_eye
  # 2: right_eye
  # 3: left_ear
  # 4: right_ear
  # 5: left_shoulder
  # 6: right_shoulder
  # 7: left_elbow
  # 8: right_elbow
  # 9: left_wrist
  # 10: right_wrist
  # 11: left_hip
  # 12: right_hip
  # 13: left_knee
  # 14: right_knee
  # 15: left_ankle
  # 16: right_ankle
# ]

LABELS_TRAIN_DIR = '/datatmp2/joan/tfg_joan/LSTM_dataset/train/labels'

CLASSES = ['bench_press', 'deadlift', 'squat', 'pull_up']

MODEL_SAVE_PATH = '/datatmp2/joan/tfg_joan/models_LSTM/LSTM_RepCount3.pth'

BATCH_SIZE = 8
SEQ_LEN = 50
EPOCHS = 60

def normalize_keypoints(coords):
    #Es normalitzen les coordenades respecte al tors
    #D'aquesta manera es fa invariant a la posició de la càmera
    
    left_shoulder, right_shoulder = 5, 6
    
    torso = (coords[:, left_shoulder] + coords[:, right_shoulder]) / 2.0  # Pixel centre tors
    torso_dist = np.linalg.norm(coords[:, left_shoulder] - coords[:, right_shoulder], axis=1)
    
    # Evitar divisió per zero
    if np.any(torso_dist == 0):
        torso_dist[torso_dist == 0] = 1.0  

    # Centrar i normalitzar les coordenades
    coords_centered = coords - torso[:, None, :]
    coords_normalized = coords_centered / torso_dist[:, None, None]

    return coords_normalized

def frame_to_feature(coords_frame):
    #Converteix les coordenades d'un frame a un vector de característiques
    # (13,2) -> (26,)
    return coords_frame.flatten()  # 26-d vector

def seq_to_features(coords_seq):
    # Converteix una seqüència de coordenades a una seqüència de vectors de característiques
    # A banda de la posició dels keypoints s'afegeix la diferencia entre el frame anterior, 
    # cosa que afegeix la informació de moviment i permet determinar l'exercici
    # (T,13,2) -> (T,52)
    
    # T = Nombre de imatges per video
    
    # Normalitzar coordenades
    coords_normalized = normalize_keypoints(coords_seq)  # (T,13,2)
    
    # Convertir cada frame a vector de característiques
    feats = np.array([frame_to_feature(frame) for frame in coords_normalized])  # (T,26)
    
    # Calcular velocitat com a diferència entre frames consecutius
    vel = np.vstack([np.zeros((1, feats.shape[1])), np.diff(feats, axis=0)])  # (T,26)
    
    # Concatenar posició i velocitat
    feats_all = np.concatenate([feats, vel], axis=1)  # (T,52)
    
    return feats_all

class KeyPointSequenceDataset(Dataset):
    def __init__(self, X_coords, y_labels, seq_len=150):
        self.X_coords = X_coords
        self.y = y_labels
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.X_coords)
    
    # Carreguem seqüències de màxim 200 frames, per no tenir problemes de memòria
    # i per no agafar totes les seqüències des del principi, sinó que agafem un clip aleatori
    def __getitem__(self, idx):
        arr = self.X_coords[idx]
        feats = seq_to_features(arr)  # (T, F)
        T, F = feats.shape
        start = 0
        if T >= self.seq_len:
            start = np.random.randint(0, T - self.seq_len + 1)
            feats = feats[start:start+self.seq_len]
            length = self.seq_len
        else:
            pad = np.zeros((self.seq_len - T, F), dtype=feats.dtype)
            feats = np.vstack([feats, pad])
            length = T

        return torch.tensor(feats, dtype=torch.float32), length, torch.tensor(self.y[idx], dtype=torch.long)
    
def collate_fn(batch):
    
    # Es retorna les seqüències en batches per a l'entrada del model
    Xs, lengths, ys = zip(*batch)
    X = torch.stack(Xs)  # (B, seq_len, F)
    lengths = torch.tensor(lengths, dtype=torch.long)
    ys = torch.stack(ys)
    return X, lengths, ys

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=52, hidden_size=256, num_layers=2, num_classes=4, dropout=0.4, num_directions=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        self.num_directions = num_directions

    def forward(self, x, lengths=None):
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
        else:
            out, (h_n, c_n) = self.lstm(x)
        if self.num_directions == 2:
            h_forward = h_n[-2,:,:]
            h_backward = h_n[-1,:,:]
            h = torch.cat([h_forward, h_backward], dim=1)  # (B, hidden_size*2)
        else:
            h = h_n[-1]
        logits = self.fc(h)
        return logits
    
def train_epoch(model, dataloader, opt, criterion, device):
    model.train()
    losses = []
    preds, trues = [], []
    for X, lengths, y in dataloader:
        X, y, lengths = X.to(device), y.to(device), lengths.to(device)
        opt.zero_grad()
        logits = model(X, lengths)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return np.mean(losses), accuracy_score(trues, preds), f1_score(trues, preds, average='macro')
        
def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=None, best_metric=None, extra=None):

    tmp_path = path + ".tmp"
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "extra": extra,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    # Guardar en archivo temporal y mover (reduce riesgo de fichero corrupto)
    torch.save(payload, tmp_path)
    shutil.move(tmp_path, path)
    print(f"Checkpoint guardado en: {path}")
            

# ---------- Ejemplo de uso ----------
if __name__ == "__main__":
    # ejemplo dummy: X_list = [np.random.rand(random_T,13,2) for ...], y_list = [...]
    
    X_list, y_list = [], []

    for class_name in CLASSES:
        for sub_dir, _, files in os.walk(os.path.join(LABELS_TRAIN_DIR, class_name)):
            video_list = []
            for file in files:
                if file.endswith('.json'):
                    if class_name in CLASSES:
                        class_idx = CLASSES.index(class_name)
                        file_path = os.path.join(sub_dir, file)
                        with open(file_path, 'r') as f:
                            
                            data = json.load(f)
                            
                            instance_info = data.get('instance_info', {})
                            
                            keypoints = instance_info[0]['keypoints']
                            
                            if keypoints is None or keypoints == np.zeros((13,2)).tolist():
                                continue
                            
                            #Eliminar cames dels keypoints
                            keypoints = [keypoints[i] for i in range(len(keypoints)) if i not in [13,14,15,16]]

                        video_list.append(np.array(keypoints).reshape(-1, 13, 2))
            if len(video_list) > 0:     
                X_list.append(np.vstack(video_list))
                y_list.append(class_idx)
                
    # Es filtren les clases per a deixar la mateixa quantitat de dades d'entrenament en cada classe

    exercises = [0,0,0,0]
    for y in y_list:
        for i in range(len(CLASSES)):
            if y == i:
                exercises[i] += 1

    print(f"Nombre d'instàncies per classe: bench_press={exercises[0]}, deadlift={exercises[1]}, squat={exercises[2]}, pull_up={exercises[3]}")
    
    min_classes = min(exercises)
    
    X_filtered_list = []
    y_filtered_list = []
    exercises = [0,0,0,0]
    
    for x, y in zip(X_list, y_list):
        for i in range(len(CLASSES)):
            if y == i and exercises[i] < min_classes:
                X_filtered_list.append(x)
                y_filtered_list.append(y)
                exercises[i] += 1
                    

    num_classes = len(CLASSES)
    train_X, val_X, train_y, val_y = train_test_split(X_filtered_list, y_filtered_list, test_size=0.2, stratify=y_filtered_list, random_state=42)

    train_ds = KeyPointSequenceDataset(train_X, train_y, seq_len=SEQ_LEN)
    val_ds   = KeyPointSequenceDataset(val_X, val_y, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_feat = seq_to_features(np.zeros((10,13,2)))
    input_size = sample_feat.shape[1]  # 13 * 2 * 2 = 52 (n. keypoints/frame * 2 coords * pos+vel)
    model = LSTMClassifier(input_size=input_size, hidden_size=256, num_layers=2, num_classes=num_classes, num_directions=1).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_f1 = 0.0
    
    for epoch in range(1, EPOCHS+1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, opt, criterion, device)
        print(f"Epoch {epoch}: loss={train_loss:.4f} acc={train_acc:.3f} f1={train_f1:.3f}")
        
        if train_f1 > best_f1:
            best_f1 = train_f1
            save_checkpoint(MODEL_SAVE_PATH, model, opt, scheduler=None, epoch=epoch, best_metric=best_f1)