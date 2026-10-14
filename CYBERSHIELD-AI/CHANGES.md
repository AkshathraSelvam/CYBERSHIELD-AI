# CyberShield-AI: changes in this version

## Notebook: notebooks/02_data_preprocessing.ipynb

1. Removed the old binary-only split and its duplicate scaler, encoder and
   save cells. The single split (train_test_split on X, y_binary and
   y_multiclass together, stratified on y_multiclass) is now the only split.
2. Logistic regression evaluation: the classification report now uses
   ["ATTACK", "BENIGN"] instead of an undefined variable (fixes NameError).
3. Removed the ROC-AUC cell that gave the wrong value (0.0188). The correct
   ROC-AUC cell remains.
4. Decision Tree binary metrics file: ROC-AUC is written from the computed
   variable, not typed in.
5. Binary model comparison table: all values come from computed variables.
6. Multiclass model comparison table: all values come from computed variables.
7. Risk pipeline (threat events and risk scoring) is now gated:
   binary Random Forest first (unscaled test features), then the multiclass
   Decision Tree only on rows flagged ATTACK. BENIGN rows get the label BENIGN.
8. Attack_Confidence: probability of the predicted attack class on attack
   rows (0 to 1). BENIGN rows are 0.
9. Anomaly bonus (+15) is added only to rows predicted as ATTACK. BENIGN
   rows always have a bonus of 0.
10. All saved outputs were cleared. Run All regenerates everything.

## App: app.py

11. Feature-importance chart uses Streamlit's native bar chart. The
    Matplotlib chart (white background on the dark theme) is removed.
12. Ground-truth check: when the uploaded file has a Label column, the app
    shows accuracy, precision, recall, F1 and a confusion matrix for binary
    detection, computed on the same rows that survived cleaning.
13. Results are tied to the file they were made from. Uploading a new file
    clears the old results, and the results header shows the file name.
14. Only the models the app uses are loaded: scaler, both label encoders,
    feature names, binary Random Forest, multiclass Decision Tree and
    Isolation Forest.
15. Negative-value rule matches the notebook: only the 11 listed features
    are checked. Init_Win_bytes_forward and Init_Win_bytes_backward may be -1.

## Before you cross-check

- The models and results folders are empty on purpose. Run notebook 02
  from top to bottom (Restart kernel, Run All) to create them.
- The notebook was not executed while preparing this zip. Its numbers are
  final only after your run.
