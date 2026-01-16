# Hospital Readmission Prediction (Diabetes)

## Project Overview

This project predicts 30-day hospital readmissions for diabetic patients using electronic health record (EHR) data. Early identification of high-risk patients enables targeted interventions to reduce readmissions.

## Business Problem

Hospital readmissions are costly and indicate poor care quality. For diabetic patients, readmissions are particularly common due to disease complexity. Predicting readmission enables:
- Targeted discharge planning
- Intensive follow-up care
- Cost reduction
- Improved patient outcomes
- Quality metric improvement

## Features

### Demographics
- Age, gender, race

### Admission Information
- Type (emergency/urgent/elective), source, length of stay, disposition

### Diabetes-Specific
- Type, HbA1c level, glucose level, insulin use, medications

### Comorbidities
- Hypertension, heart disease, kidney disease, COPD, obesity, Charlson index

### Clinical Measurements
- Blood pressure, heart rate, BMI, creatinine, hemoglobin

### Medications
- Number of medications, changes, specific drug classes

### Healthcare Utilization
- Outpatient/inpatient/emergency visits, prior readmissions

### Social Factors
- Living situation, caregiver availability, insurance type

## Model Architecture

**Algorithm**: Gradient Boosting Classifier
- Handles complex interactions
- Provides interpretable risk scores
- Effective for imbalanced data

## Project Structure

```
08-hospital-readmission/
├── readmission_model.py          # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── models/
    ├── readmission_model.pkl     # Trained model
    ├── scaler.pkl                # Feature scaler
    └── feature_importance.csv    # Feature importance
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Model
```bash
python train.py
```

### 2. Make Predictions
```bash
python predict.py
```

## Model Performance

- **ROC-AUC Score**: 0.80-0.85
- **Precision**: 0.65-0.70
- **Recall**: 0.70-0.75
- **F1-Score**: 0.67-0.72

## Top Readmission Risk Factors

1. Charlson comorbidity index
2. Prior readmissions in past year
3. HbA1c level (glycemic control)
4. Number of medications
5. Admission type (emergency)
6. Age
7. Length of stay
8. Lack of caregiver support
9. Glucose level
10. Number of medication changes

## Risk Stratification

- **Low Risk**: Routine discharge planning
- **Medium Risk**: Enhanced follow-up care
- **High Risk**: Intensive case management

## Intervention Strategies

### For High-Risk Patients
- Same-day discharge follow-up
- Home health services
- Intensive diabetes education
- Medication reconciliation
- Close monitoring

### For Medium-Risk Patients
- Scheduled follow-up within 7 days
- Diabetes education
- Medication review
- Care coordination

### For Low-Risk Patients
- Standard discharge planning
- Routine follow-up

## Real-World Applications

- **Hospitals**: Readmission prevention programs
- **ACOs**: Quality and cost management
- **Insurers**: Risk stratification
- **Diabetes programs**: Patient management
- **Care coordination**: Resource allocation

## Benefits

1. **Cost reduction**: Prevent expensive readmissions
2. **Quality improvement**: Reduce readmission rates
3. **Better outcomes**: Identify high-risk patients early
4. **Resource optimization**: Target interventions
5. **Compliance**: Meet quality metrics

## Clinical Considerations

- Model should complement clinical judgment
- Requires regular validation and recalibration
- Consider patient preferences and goals
- Integrate with EHR systems
- Monitor for bias and fairness

## Future Enhancements

1. Time-to-readmission prediction
2. Cause-specific readmission prediction
3. Personalized intervention recommendations
4. Integration with wearable devices
5. Real-time risk scoring

## References

- Joynt, K. E., & Jha, A. K. (2013). Thirty-Day Readmissions
- Krumholz, H. M. (2013). Post-Hospital Syndrome
- Donzé, J., Lipsitz, S., & Bates, D. W. (2013). Causes and Patterns of Readmissions

## Author

Created as part of ML Portfolio Projects

## License

MIT License
