"""
extract_features_fast.py

Optimized Multimodal Feature Extraction Pipeline for CREMA-D.

Optimizations Applied:
1. Parallel Processing: Uses ProcessPoolExecutor to utilize all available CPU cores.
2. Fast Pitch Estimation: Replaces slow librosa.pyin with librosa.yin (10-20x speedup).
3. Frame Subsampling: Processes every N-th video frame (stride=3) to reduce MediaPipe computational overhead by ~66-75%.
"""

import os
import cv2
import glob
import numpy as np
import librosa
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# Target emotion mapping: 3-class valence scheme
EMOTION_MAP = {
    'HAP': 0,  # Positive (Happy)
    'SAD': 1,  # Negative (Sad)
    'ANG': 1,  # Negative (Anger)
    'FEA': 1,  # Negative (Fear)
    'DIS': 1,  # Negative (Disgust)
    'NEU': 2   # Neutral (Neutral)
}

def extract_audio_features_fast(audio_path):
    """
    Extracts 32-dim static acoustic summary vector.
    Uses librosa.yin for fast fundamental frequency tracking.
    """
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Optimized Pitch (F0) using YIN algorithm (10-20x faster than pyin)
    try:
        f0 = librosa.yin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = float(np.nanmean(f0)) if np.any(~np.isnan(f0)) else 0.0
    except Exception:
        f0_mean = 0.0

    # Energy, Centroid, ZCR
    rms_mean = float(np.mean(librosa.feature.rms(y=y)))
    centroid_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    zcr_mean = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    
    # MFCCs (13 means + 13 variances)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)
    mfcc_vars = np.var(mfcc, axis=1)
    
    # Combine into 32-dim vector
    audio_vec = np.hstack([rms_mean, f0_mean, centroid_mean, zcr_mean, mfcc_means, mfcc_vars])
    return audio_vec

def extract_visual_features_fast(video_path, mp_face_mesh, frame_stride=3):
    """
    Extracts 128-dim static landmark geometry & temporal summary vector.
    Applies frame subsampling (frame_stride) to accelerate MediaPipe processing.
    """
    cap = cv2.VideoCapture(video_path)
    frame_landmarks = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        # Skip frames to reduce video extraction overhead
        if frame_count % frame_stride != 0:
            continue
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Extract (x, y, z) coordinates for 468 landmarks -> 1404 values per frame
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]).flatten()
            frame_landmarks.append(landmarks)
            
    cap.release()
    
    if len(frame_landmarks) == 0:
        return np.zeros(128)
        
    frame_landmarks = np.array(frame_landmarks)
    
    # Calculate summary statistics across time
    means = np.mean(frame_landmarks, axis=0)
    stds = np.std(frame_landmarks, axis=0)
    
    # Frame-to-frame velocity
    velocity = np.diff(frame_landmarks, axis=0) if len(frame_landmarks) > 1 else np.zeros_like(frame_landmarks)
    max_velocity = np.max(np.abs(velocity), axis=0)
    
    # Subsample top dynamic features down to fixed 128 dimensions
    summary_vector = np.hstack([means[:48], stds[:48], max_velocity[:32]])
    return summary_vector

def process_single_clip(filepath, audio_dir, frame_stride=3):
    """
    Worker function to process a single audio/video pair in parallel.
    """
    filename = os.path.basename(filepath)
    parts = filename.split('_')
    
    if len(parts) < 3 or parts[2] not in EMOTION_MAP:
        return None
        
    actor_id = int(parts[0])
    emotion_code = parts[2]
    audio_path = os.path.join(audio_dir, filename.replace('.flv', '.wav'))
    
    if not os.path.exists(audio_path):
        return None
        
    try:
        # Re-initialize MediaPipe inside each worker process
        mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        
        aud_feat = extract_audio_features_fast(audio_path)
        vis_feat = extract_visual_features_fast(filepath, mp_face_mesh, frame_stride=frame_stride)
        
        mp_face_mesh.close()
        return (aud_feat, vis_feat, EMOTION_MAP[emotion_code], actor_id)
    except Exception as e:
        return None

def process_cremad_dataset_fast(data_dir, audio_dir, output_file='cremad_features.npz',
                                max_workers=None, frame_stride=3):
    """
    Main extraction loop leveraging multi-core parallel execution.
    """
    video_files = glob.glob(os.path.join(data_dir, "*.flv"))
    if not video_files:
        video_files = glob.glob(os.path.join(data_dir, "*.mp4"))
        
    num_files = len(video_files)
    print(f"Found {num_files} clips in {data_dir}.")
    
    if max_workers is None:
        max_workers = os.cpu_count() or 4
        
    print(f"Starting parallel processing on {max_workers} CPU cores (frame stride = {frame_stride})...")
    
    X_audio, X_video, Y_labels, actor_ids = [], [], [], []
    completed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_clip, f, audio_dir, frame_stride): f
            for f in video_files
        }
        
        for future in as_completed(futures):
            res = future.result()
            completed += 1
            if completed % 500 == 0 or completed == num_files:
                print(f"Processed {completed}/{num_files} clips...")
                
            if res is not None:
                aud, vis, y, act = res
                X_audio.append(aud)
                X_video.append(vis)
                Y_labels.append(y)
                actor_ids.append(act)
                
    np.savez(output_file, 
             X_audio=np.array(X_audio), 
             X_video=np.array(X_video), 
             Y=np.array(Y_labels), 
             actors=np.array(actor_ids))
             
    print(f"Extraction complete! Successfully processed {len(Y_labels)} clips.")
    print(f"Features saved to: {output_file}")

if __name__ == "__main__":
    DATASET_DIR = "./data/VideoFLV"
    AUDIO_DIR = "./data/AudioWAV"
    
    if os.path.exists(DATASET_DIR) and os.path.exists(AUDIO_DIR):
        process_cremad_dataset_fast(data_dir=DATASET_DIR, audio_dir=AUDIO_DIR)
    else:
        print(f"Dataset directory not found: '{DATASET_DIR}' or '{AUDIO_DIR}'.")
