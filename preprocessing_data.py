import os
import cv2
import glob
import numpy as np
import pandas as pd
import librosa
import mediapipe as mp

# Target emotion mapping: 3-class valence scheme
EMOTION_MAP = {
    'HAP': 0,  # Positive (Happy)
    'SAD': 1,  # Negative (Sad)
    'ANG': 1,  # Negative (Anger)
    'FEA': 1,  # Negative (Fear)
    'DIS': 1,  # Negative (Disgust)
    'NEU': 2   # Neutral (Neutral)
}

def extract_audio_features(audio_path):
    """Extracts 32-dim static acoustic summary vector."""
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Pitch (F0)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0
    
    # Energy, Centroid, ZCR
    rms_mean = np.mean(librosa.feature.rms(y=y))
    centroid_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
    
    # MFCCs (13 means + 13 variances)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_vars = np.var(mfcc, axis=1)
    
    # Combine into 32-dim vector
    audio_vec = np.hstack([rms_mean, f0_mean, centroid_mean, zcr_mean, mfcc_means, mfcc_vars])
    return audio_vec

def extract_visual_features(video_path, mp_face_mesh):
    """Extracts 128-dim static landmark geometry & temporal summary vector."""
    cap = cv2.VideoCapture(video_path)
    frame_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Flatten (x, y, z) for 468 landmarks -> 1404 features per frame
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]).flatten()
            frame_landmarks.append(landmarks)
            
    cap.release()
    
    if len(frame_landmarks) == 0:
        return np.zeros(128)  # Fallback if face lost
        
    frame_landmarks = np.array(frame_landmarks)  # Shape: (T_frames, 1404)
    
    # Calculate global temporal statistics over time
    means = np.mean(frame_landmarks, axis=0)
    stds = np.std(frame_landmarks, axis=0)
    
    # Frame-to-frame velocity
    velocity = np.diff(frame_landmarks, axis=0) if len(frame_landmarks) > 1 else np.zeros_like(frame_landmarks)
    max_velocity = np.max(np.abs(velocity), axis=0)
    
    # Subsample top dynamic features down to fixed 128 dimensions
    summary_vector = np.hstack([means[:48], stds[:48], max_velocity[:32]])
    return summary_vector

def process_cremad_dataset(data_dir, output_file='cremad_features.npz'):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    X_audio, X_video, Y_labels, actor_ids = [], [], [], []
    video_files = glob.glob(os.path.join(data_dir, "*.flv"))
    
    print(f"Found {len(video_files)} clips. Starting feature extraction...")
    
    for filepath in video_files:
        filename = os.path.basename(filepath)
        parts = filename.split('_')
        
        actor_id = int(parts[0])
        emotion_code = parts[2]
        
        if emotion_code not in EMOTION_MAP:
            continue
            
        audio_path = filepath.replace('.flv', '.wav')
        if not os.path.exists(audio_path):
            continue
            
        try:
            aud_feat = extract_audio_features(audio_path)
            vis_feat = extract_visual_features(filepath, mp_face_mesh)
            
            X_audio.append(aud_feat)
            X_video.append(vis_feat)
            Y_labels.append(EMOTION_MAP[emotion_code])
            actor_ids.append(actor_id)
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            
    np.savez(output_file, 
             X_audio=np.array(X_audio), 
             X_video=np.array(X_video), 
             Y=np.array(Y_labels), 
             actors=np.array(actor_ids))
    print(f"Extraction complete! Saved to {output_file}")

if __name__ == "__main__":
    # Update path to your local CREMA-D directory
    process_cremad_dataset(data_dir="./CREMA-D")