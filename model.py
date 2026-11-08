import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- 1. Dataset Wrapper ---
class CREMADataset(Dataset):
    def __init__(self, x_audio, x_video, y):
        self.x_audio = torch.tensor(x_audio, dtype=torch.float32)
        self.x_video = torch.tensor(x_video, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x_audio[idx], self.x_video[idx], self.y[idx]

# --- 2. Neural Architecture ---
class AudioEncoder(nn.Module):
    def __init__(self, input_dim=32, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class VisualEncoder(nn.Module):
    def __init__(self, input_dim=128, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class LateFusionModel(nn.Module):
    def __init__(self, audio_dim=32, video_dim=128, num_classes=3):
        super().__init__()
        self.audio_encoder = AudioEncoder(audio_dim, num_classes)
        self.visual_encoder = VisualEncoder(video_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))  # Sigmoid(0) = 0.5

    def forward(self, x_audio, x_video):
        logits_audio = self.audio_encoder(x_audio)
        logits_video = self.visual_encoder(x_video)
        
        prob_audio = self.softmax(logits_audio)
        prob_video = self.softmax(logits_video)
        
        alpha = torch.sigmoid(self.alpha_raw)
        p_final = alpha * prob_audio + (1.0 - alpha) * prob_video
        return p_final, alpha, prob_audio, prob_video

# --- 3. Permutation Feature Importance Helper ---
def compute_permutation_importance(model, val_loader, baseline_acc, audio_dim=32, video_dim=128):
    """Calculates accuracy drop when shuffling each feature column independently."""
    model.eval()
    
    # Collect all validation data into memory tensors
    all_a, all_v, all_y = [], [], []
    with torch.no_grad():
        for ba, bv, by in val_loader:
            all_a.append(ba)
            all_v.append(bv)
            all_y.append(by)
            
    all_a = torch.cat(all_a, dim=0)
    all_v = torch.cat(all_v, dim=0)
    all_y = torch.cat(all_y, dim=0).numpy()

    audio_importances = np.zeros(audio_dim)
    video_importances = np.zeros(video_dim)

    with torch.no_grad():
        # Permute Audio Features
        for i in range(audio_dim):
            a_perm = all_a.clone()
            a_perm[:, i] = a_perm[torch.randperm(a_perm.size(0)), i]
            p_final, _, _, _ = model(a_perm, all_v)
            preds = torch.argmax(p_final, dim=1).numpy()
            acc_perm = accuracy_score(all_y, preds)
            audio_importances[i] = baseline_acc - acc_perm

        # Permute Video Features
        for j in range(video_dim):
            v_perm = all_v.clone()
            v_perm[:, j] = v_perm[torch.randperm(v_perm.size(0)), j]
            p_final, _, _, _ = model(all_a, v_perm)
            preds = torch.argmax(p_final, dim=1).numpy()
            acc_perm = accuracy_score(all_y, preds)
            video_importances[j] = baseline_acc - acc_perm

    return audio_importances, video_importances

# --- 4. Main Evaluation & Training Pipeline ---
def train_evaluate_ablation(data_path='cremad_features.npz'):
    data = np.load(data_path)
    X_audio = data['X_audio']
    X_video = data['X_video']
    Y = data['Y']
    actors = data['actors']

    gkf = GroupKFold(n_splits=5)
    
    # Per-fold accuracy tracking for Ablation & Hypothesis Testing
    fold_acc_fusion = []
    fold_acc_audio_only = []
    fold_acc_video_only = []
    
    audio_importance_folds = []
    video_importance_folds = []

    print("==================================================================")
    print(f"Starting 5-Fold GroupKFold Cross-Validation on CREMA-D ({len(Y)} samples)")
    print("==================================================================\n")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_audio, Y, groups=actors)):
        scaler_a, scaler_v = StandardScaler(), StandardScaler()

        X_a_train = scaler_a.fit_transform(X_audio[train_idx])
        X_a_val = scaler_a.transform(X_audio[val_idx])

        X_v_train = scaler_v.fit_transform(X_video[train_idx])
        X_v_val = scaler_v.transform(X_video[val_idx])

        train_ds = CREMADataset(X_a_train, X_v_train, Y[train_idx])
        val_ds = CREMADataset(X_a_val, X_v_val, Y[val_idx])

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = LateFusionModel(audio_dim=X_audio.shape[1], video_dim=X_video.shape[1])
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        # Training Phase
        for epoch in range(40):
            model.train()
            for ba, bv, by in train_loader:
                optimizer.zero_grad()
                p_final, _, _, _ = model(ba, bv)
                loss = criterion(torch.log(p_final + 1e-8), by)
                loss.backward()
                optimizer.step()

        # Ablation Evaluation Phase
        model.eval()
        preds_fusion, preds_audio, preds_video, targets = [], [], [], []
        
        with torch.no_grad():
            for ba, bv, by in val_loader:
                p_final, _, p_audio, p_video = model(ba, bv)
                
                preds_fusion.extend(torch.argmax(p_final, dim=1).cpu().numpy())
                preds_audio.extend(torch.argmax(p_audio, dim=1).cpu().numpy())
                preds_video.extend(torch.argmax(p_video, dim=1).cpu().numpy())
                targets.extend(by.cpu().numpy())

        acc_f = accuracy_score(targets, preds_fusion)
        acc_a = accuracy_score(targets, preds_audio)
        acc_v = accuracy_score(targets, preds_video)

        fold_acc_fusion.append(acc_f)
        fold_acc_audio_only.append(acc_a)
        fold_acc_video_only.append(acc_v)

        # Feature Importance Phase
        imp_a, imp_v = compute_permutation_importance(
            model, val_loader, baseline_acc=acc_f, 
            audio_dim=X_audio.shape[1], video_dim=X_video.shape[1]
        )
        audio_importance_folds.append(imp_a)
        video_importance_folds.append(imp_v)

        print(f"Fold {fold+1} Accuracies -> Fusion: {acc_f:.4f} | Audio-Only: {acc_a:.4f} | Video-Only: {acc_v:.4f}")

    # --- Statistical Analysis & Results Reporting ---
    mean_f, mean_a, mean_v = np.mean(fold_acc_fusion), np.mean(fold_acc_audio_only), np.mean(fold_acc_video_only)
    best_unimodal = max(mean_a, mean_v)
    best_unimodal_scores = fold_acc_audio_only if mean_a > mean_v else fold_acc_video_only
    best_unimodal_name = "Audio-Only" if mean_a > mean_v else "Video-Only"

    # Paired t-Test (Fusion vs. Best Unimodal Baseline)
    t_stat, p_value = stats.ttest_rel(fold_acc_fusion, best_unimodal_scores)

    print("\n=================== ABLATION STUDY RESULTS ===================")
    print(f"Late Fusion (Multimodal) Mean Acc : {mean_f:.4f} (+/- {np.std(fold_acc_fusion):.4f})")
    print(f"Audio-Only Modality Mean Acc      : {mean_a:.4f} (+/- {np.std(fold_acc_audio_only):.4f})")
    print(f"Video-Only Modality Mean Acc      : {mean_v:.4f} (+/- {np.std(fold_acc_video_only):.4f})")
    print("==============================================================")

    print("\n============== STATISTICAL HYPOTHESIS TESTING ==============")
    print(f"Test: Paired t-Test (Late Fusion vs. Best Unimodal: {best_unimodal_name})")
    print(f"t-Statistic : {t_stat:.4f}")
    print(f"p-Value     : {p_value:.6f}")
    if p_value < 0.05:
        print("Conclusion  : Reject Null Hypothesis (H0). Multimodal fusion yields a STATISTICALLY SIGNIFICANT performance improvement over unimodal baselines.")
    else:
        print("Conclusion  : Fail to Reject Null Hypothesis (H0). Difference between fusion and unimodal baselines is not statistically significant.")
    print("==============================================================")

    # Feature Importance Top Ranks
    avg_audio_imp = np.mean(audio_importance_folds, axis=0)
    avg_video_imp = np.mean(video_importance_folds, axis=0)

    print("\n================ FEATURE IMPORTANCE (TOP 5) ================")
    top_audio_idx = np.argsort(avg_audio_imp)[::-1][:5]
    top_video_idx = np.argsort(avg_video_imp)[::-1][:5]
    
    print("Top Audio Features (Index: Mean Accuracy Drop):")
    for idx in top_audio_idx:
        print(f"  - Audio Feature [{idx:02d}]: {avg_audio_imp[idx]:.5f}")

    print("\nTop Visual Features (Index: Mean Accuracy Drop):")
    for idx in top_video_idx:
        print(f"  - Video Feature [{idx:03d}]: {avg_video_imp[idx]:.5f}")
    print("==============================================================")

if __name__ == "__main__":
    train_evaluate_ablation()