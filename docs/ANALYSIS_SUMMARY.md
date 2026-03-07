# Model Analysis Summary

## Baseline Comparison Results

### 1. Simple Rule (delta_rating > 0 => +1)
- **Accuracy: 39.79%**
- WRONG interpretation! This is worse than random
- The correct rule is inverted (see below)

### 1b. Simple Rule CORRECTED (delta_rating > 0 => -1)
- **Accuracy: 60.21%**
- Correct interpretation: positive delta means Team 1 is stronger
- This is actually a strong baseline!

### 2. Model with ONLY delta_rating
- **Accuracy: 60.34%**
- Only +0.13% better than simple rule
- Shows SVM can find slight non-linear patterns

### 3. Full Model (all features)
- **Accuracy: 60.47%**
- Only +0.26% over corrected simple rule
- Suggests most features add minimal value

## Ablation Study (Feature Importance)

### Features that HURT performance:
- **map_id**: Removing it IMPROVES accuracy by 0.52% (60.98% vs 60.47%)
  - Likely overfitting to specific maps in training data
  - Map preferences don't generalize well

### Features with NO impact:
- **delta_map_wr_windowed**: Removing has 0.00% impact
  - Windowed map winrate difference is redundant or noisy
  - Already captured by other features

### Features that HELP:
- **h2h features (h2h_shrunk, h2h_count)**: Removing drops accuracy by 0.90%
  - Head-to-head history is valuable
  - Direct matchup data provides unique signal

## Optimized Model

### Configuration:
- **Features**: delta_rating, map_missing_flag, LAN_flag, h2h_shrunk, h2h_count
- **Kernel**: Linear
- **C**: 0.5
- **Accuracy**: 61.63%

### Performance Improvement:
- vs Simple Rule (WRONG): +21.84%
- vs Simple Rule (CORRECT): +1.42%
- vs Full Model: +1.16%
- vs Delta Rating Only: +1.29%

**Key Insight:** The corrected simple rule (60.21%) is surprisingly strong! ML models only add marginal improvement.

### Confusion Matrix Analysis:
```
                Predicted
                -1    +1
Actual  -1     184   194
        +1     103   293
```

- **True Negatives**: 184 (48.7% of actual -1)
- **False Positives**: 194 (51.3% of actual -1)
- **False Negatives**: 103 (26.0% of actual +1)
- **True Positives**: 293 (74.0% of actual +1)

**Observation**: Model is biased toward predicting +1 (winner)
- Better recall for class +1 (74%) vs class -1 (49%)
- This might be acceptable if false negatives are more costly

## Key Insights

1. **Less is More**: Removing features improved performance
   - Feature selection is critical
   - More data ≠ better predictions

2. **Rating Difference is King**: delta_rating provides most signal
   - Other features add marginal value
   - Complex features (map stats) don't help

3. **H2H Matters**: Direct matchup history is valuable
   - Personal dynamics matter beyond ratings
   - Small but consistent improvement

4. **Map Features Failed**:
   - map_id: overfitting to training maps
   - delta_map_wr_windowed: redundant or noisy
   - Consider removing map features entirely

## Recommendations

### Immediate Actions:
1. ✅ Use optimized model (5 features, linear kernel, C=0.5)
2. ✅ Remove map_id and delta_map_wr_windowed from pipeline
3. Consider class weights if false negatives are costly

### Future Improvements:

#### 1. Feature Engineering
- Interaction terms: delta_rating × h2h_shrunk
- Non-linear transforms: delta_rating², log(|delta_rating|)
- Recent form: last N games performance
- Momentum indicators

#### 2. Try Other Models
- Logistic Regression (interpretable baseline)
- Random Forest (handles non-linearity)
- XGBoost (often best for tabular data)
- Neural Network (if more data available)

#### 3. Data Quality
- Investigate why simple rule fails (39.79%)
- Check for data leakage or label errors
- Analyze misclassified examples
- Consider temporal validation (not just random split)

#### 4. Ensemble Methods
- Combine multiple models
- Voting classifier
- Stacking with meta-learner

#### 5. Calibration
- Current model outputs class labels
- Probability calibration for betting/ranking
- Platt scaling or isotonic regression

## Conclusion

The optimized model achieves **61.63% accuracy** by:
- Removing harmful features (map_id)
- Removing useless features (delta_map_wr_windowed)
- Keeping core predictive features (delta_rating, h2h, flags)
- Using linear SVM with C=0.5

This represents a **1.16% improvement** over the full model and demonstrates that careful feature selection outperforms naive "use everything" approaches.

The model is saved in `final_model.pkl` and ready for deployment.
