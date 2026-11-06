# Five-class fixed XGBoost transfer and labelled adaptation

87 subjects; 8,700 trials; 59 combined_unique features; labels BL1/PA1/PA2/PA3/PA4. Seed 20260909.

Cross reproduces the original five subject-grouped folds and fixed benchmark configuration. No tuning or model selection was performed. All median imputers were fit on training samples only.
Original cross prediction mismatches: 0. Pooled cross macro F1: 0.305227.

Within uses 5-fold stratified CV repeated 10 times, 80 training trials per subject per fit. Primary comparisons average repeat metrics within subject, then average subjects. Probability-averaged within OOF metrics are secondary and use an ensemble across repeats. Cross training uses thousands of trials: the gap is not a pure subject-shift effect.

Adaptation uses only the held-out subject calibration labels to fit six strongly regularized score-calibration parameters on frozen population XGBoost outputs. k is PER CLASS (total 5k). This is score calibration, not retraining or fine-tuning the trees. Evaluation excludes calibration trials; population and adapted predictions use identical evaluation samples. Budgets are nested per repeat. No target evaluation labels enter fitting.

## Subject-mean macro F1 (95% subject bootstrap CI)

- adapted, k/class=1: 0.2939 [0.2795, 0.3084]
- adapted, k/class=2: 0.2967 [0.2816, 0.3128]
- adapted, k/class=5: 0.3040 [0.2883, 0.3193]
- adapted, k/class=10: 0.3067 [0.2912, 0.3224]
- cross, k/class=0: 0.2906 [0.2754, 0.3057]
- population_matched, k/class=1: 0.2906 [0.2762, 0.3048]
- population_matched, k/class=2: 0.2904 [0.2762, 0.3048]
- population_matched, k/class=5: 0.2900 [0.2754, 0.3048]
- population_matched, k/class=10: 0.2866 [0.2713, 0.3010]
- within, k/class=0: 0.3666 [0.3422, 0.3932]
- within_averaged_probability, k/class=0: 0.3744 [0.3483, 0.4029]

## Paired macro F1 differences

- within minus cross, k/class=0: +0.0760 [+0.0561, +0.0976]
- adapted minus population_matched, k/class=1: +0.0032 [+0.0012, +0.0054]
- adapted minus population_matched, k/class=2: +0.0063 [+0.0033, +0.0092]
- adapted minus population_matched, k/class=5: +0.0140 [+0.0100, +0.0181]
- adapted minus population_matched, k/class=10: +0.0201 [+0.0143, +0.0259]

These exploratory results are conditional on the existing features, fixed model, folds and cohort. Bootstrap intervals resample subjects after repeat averaging; they do not capture training-cohort resampling uncertainty. No significance tests or selective reporting of budgets are used. Fixed XGBoost may be unsuitable for small within-subject training sets; lack of improvement is not an information ceiling. Known BL1 recovery-period and experimental-heat-pain limitations apply. No clinical or causal interpretation is justified.
Elapsed seconds: 1146.6.
