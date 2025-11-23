
# slime mold aggregation prediction model report

## Project Overview
- **Task**: Predicting the final aggregation center of Dictyostelium cells using early time frames
- **Data**: 3 experiments (mixin44, mixin57, mixin64)
- **Feature Dimension**: 28
- **Total sample size**: 688

## Optimal Model
- **Model**: center_baseline
- **Category**: Baseline
- **Center Error**: 104.43 μm
- **AUROC**: 0.6652

## Model performance ranking
1. center_baseline (Baseline): 104.43 μm, AUROC: 0.665
2. mlp (Deep Learning): 109.79 μm, AUROC: 0.651
3. xgboost (Traditional ML): 112.87 μm, AUROC: 0.645
4. random_forest (Traditional ML): 112.87 μm, AUROC: 0.637
5. nearest_neighbor (Baseline): 112.87 μm, AUROC: 0.632
6. random_baseline (Baseline): 126.92 μm, AUROC: 0.526
7. svm (Traditional ML): 145.36 μm, AUROC: 0.491
8. ridge_regression (Traditional ML): 135.09 μm, AUROC: 0.483
9. linear_regression (Traditional ML): 189.91 μm, AUROC: 0.407
10. cnn (Deep Learning): 745.75 μm, AUROC: 0.242

## Data Segmentation
- **Training set**: 492 sample
- **Validation set**: 137 sample  
- **Test set**: 59 sample

## Experimental allocation
- **Training Experiment**: mixin44, mixin57
- **Verification Experiment**: mixin64
- **Test Experiment**: mixin64

Generation time: 2025-11-23 16:04:29.943170
