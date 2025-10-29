import torch #type: ignore
import torch.nn.functional as F #type: ignore
import numpy as np
import torch.nn as nn #type: ignore
import json
import os

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=68, hidden_size=256, num_layers=2, num_classes=4, dropout=0.4, num_directions=1):
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
    # (17,2) -> (34,)
    return coords_frame.flatten()  # 34-d vector
    
def seq_to_features(coords_seq):
    # Converteix una seqüència de coordenades a una seqüència de vectors de característiques
    # A banda de la posició dels keypoints s'afegeix la diferencia entre el frame anterior, 
    # cosa que afegeix la informació de moviment i permet determinar l'exercici
    # (T,17,2) -> (T,68)
    
    # T = Nombre de imatges per video
    
    # Normalitzar coordenades
    coords_normalized = normalize_keypoints(coords_seq)  # (T,17,2)
    
    # Convertir cada frame a vector de característiques
    feats = np.array([frame_to_feature(frame) for frame in coords_normalized])  # (T,34)
    
    # Calcular velocitat com a diferència entre frames consecutius
    vel = np.vstack([np.zeros((1, feats.shape[1])), np.diff(feats, axis=0)])  # (T,34)
    
    # Concatenar posició i velocitat
    feats_all = np.concatenate([feats, vel], axis=1)  # (T,68)
    
    return feats_all

def predict_video(video_coords, model, device, class_names=None, seq_len=50):
    
    feats = seq_to_features(video_coords)
    
    T, feat_dim  = feats.shape
    
    #Es separen els videos en clips de seq_len
    clips = []
    
    if T <= seq_len:
        pad = np.zeros((seq_len - T, feat_dim), dtype=feats.dtype)
        clip = np.vstack([feats, pad])
        clips.append(clip)
    else:
        for start in range(0, T - seq_len + 1, seq_len):
            clip = feats[start:start + seq_len]
            clips.append(clip)
        # Es descarten els clips si són massa curts
        if (T - seq_len) % seq_len < 80: 
            clip = feats[-seq_len:]
            clips.append(clip)

    clips = np.stack(clips)
    X = torch.tensor(clips, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        lengths = torch.full((X.shape[0],), seq_len, dtype=torch.long).to(device)
        logits = model(X, lengths)
        
        probs = F.softmax(logits, dim=1)
        avg_probs = probs.mean(dim=0)
        
    pred_class = avg_probs.argmax().item()

    return class_names[pred_class], avg_probs.cpu().numpy()

def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location=None):

    if map_location is None:
        map_location = torch.device("cpu")
        
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        
    print(f"Checkpoint cargado desde: {path}, best_metric={ckpt.get('best_metric')})")
    
    return {"epoch": ckpt.get("epoch"), "best_metric": ckpt.get("best_metric")}

def lstm_main(args):

    if args["vel"] == False:
        input_size = args["n_keypoints"] * 2
    else:
        input_size = args["n_keypoints"] * 2 * 2  # Posició + Velocitat

    num_classes = len(args["classes"])

    model = LSTMClassifier(input_size=input_size, hidden_size=256, num_layers=2, num_classes=num_classes)

    load_checkpoint(args["model_path"], model, map_location="cuda")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pred_label, probs = predict_video(args['keypoints'], model, device, seq_len=args["seq_len"], class_names=args["classes"])

    return pred_label, probs