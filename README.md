# ISC-353-Final-Project

## Data Download Links

Audio (data.zip): https://drive.google.com/file/d/1vjsIllhIQK4jIj23_8Lkm9xBturPnAr2/view?usp=sharing
Video (archive.zip): https://www.kaggle.com/datasets/stefanogiannini/crema-d-video

## Initial Setup

Download the datasets below and place the zip and tar.gz in the root project directory before running the setup script.

## What to Run

1. setup.sh: Run this first to setup the venv and dependencies.
2. extract_features_fast.py: This is the data preprocessor.
3. model.py: This is the model.

## Directory Structure

/ISC-353-Final-Project
    data
        AudioWAV
            .....
        VideoFLV
            .....
    archive.zip
    crema_d.tar.gz
    model.py
    extract_features_fast.py
    README.md
    requirements.txt

## Results (Expected)

==================================================================
Starting 5-Fold GroupKFold Cross-Validation on CREMA-D (7441 samples)
==================================================================

Fold 1 Accuracies -> Fusion: 0.7844 | Audio-Only: 0.6719 | Video-Only: 0.7647
Fold 2 Accuracies -> Fusion: 0.8061 | Audio-Only: 0.6827 | Video-Only: 0.7824
Fold 3 Accuracies -> Fusion: 0.7851 | Audio-Only: 0.6929 | Video-Only: 0.7539
Fold 4 Accuracies -> Fusion: 0.7542 | Audio-Only: 0.6546 | Video-Only: 0.6837
Fold 5 Accuracies -> Fusion: 0.7796 | Audio-Only: 0.6694 | Video-Only: 0.7374

=================== ABLATION STUDY RESULTS ===================
Late Fusion (Multimodal) Mean Acc : 0.7819 (+/- 0.0166)
Audio-Only Modality Mean Acc      : 0.6743 (+/- 0.0129)
Video-Only Modality Mean Acc      : 0.7444 (+/- 0.0337)
==============================================================

============== STATISTICAL HYPOTHESIS TESTING ==============
Test: Paired t-Test (Late Fusion vs. Best Unimodal: Video-Only)
t-Statistic : 4.1122
p-Value     : 0.014706
Conclusion  : Reject Null Hypothesis (H0). Multimodal fusion yields a STATISTICALLY SIGNIFICANT performance improvement over unimodal baselines.
==============================================================

================ FEATURE IMPORTANCE (TOP 5) ================
Top Audio Features (Index: Mean Accuracy Drop):
  - Audio Feature [01]: 0.00552
  - Audio Feature [16]: 0.00457
  - Audio Feature [00]: 0.00431
  - Audio Feature [28]: 0.00345
  - Audio Feature [23]: 0.00321

Top Visual Features (Index: Mean Accuracy Drop):
  - Video Feature [008]: 0.18657
  - Video Feature [029]: 0.11297
  - Video Feature [035]: 0.10343
  - Video Feature [026]: 0.09134
  - Video Feature [002]: 0.09058
==============================================================