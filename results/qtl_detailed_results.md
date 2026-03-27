# Quantum Transfer Learning - Detailed Results

| Model                                | Parameters   |   Best Acc (%) |   Final Acc (%) |   Best F1 (%) |   Final Loss |   Avg Epoch (s) |   Total Time (min) |
|:-------------------------------------|:-------------|---------------:|----------------:|--------------:|-------------:|----------------:|-------------------:|
| LeNet5 Baseline (No Transfer)        | 42K          |          72.41 |           72.41 |         71.5  |       0.8286 |            43.6 |               14.5 |
| LeNet5 Original (No Standardization) | 42K          |          69.01 |           66.42 |         62.78 |       0.9936 |            44.7 |               14.9 |
| LeNet5 QTL (Frozen)                  | 42K          |          73.46 |           71.54 |         72.23 |       0.7479 |            27.8 |                9.3 |
| LeNet5 QTL (Fine-tuned)              | 42K          |          54.07 |           45.99 |         48.01 |       1.0707 |            44.9 |               15   |
| ResNet50 Source                      | 25.6M        |          96.05 |           94.14 |         95.95 |       0.4666 |            58.6 |                9.8 |

## Analysis

**Transfer Learning Benefit:** -18.34% accuracy improvement

**Parameter Reduction:** 600x fewer parameters (ResNet50 vs LeNet5)
