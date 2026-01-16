# Early Sepsis Detection

## Project Overview

This project predicts sepsis onset 6 hours before clinical manifestation using real-time time-series clinical data. Early detection enables rapid intervention and significantly improves patient outcomes.

## Business Problem

Sepsis is a life-threatening condition requiring immediate treatment. Early detection is critical:
- Mortality increases 7-10% per hour without treatment
- Early recognition improves survival rates
- Rapid antibiotic administration saves lives
- Reduces ICU stays and costs

## Features

### Vital Signs (Baseline & Current)
- Heart rate, respiratory rate, temperature
- Blood pressure (systolic/diastolic)
- Oxygen saturation

### Laboratory Values
- WBC count, hemoglobin, platelets
- Glucose, lactate, creatinine
- Bilirubin, procalcitonin

### Blood Gas Analysis
- pH, PCO2, PO2, bicarbonate

### Coagulation Studies
- INR, PT, aPTT

### Clinical Risk Factors
- Age, immunocompromised status
- Recent surgery, central line, ventilation
- Catheter, recent antibiotics

### Derived Features
- Vital sign changes over 6 hours
- Trend indicators

## Model Architecture

**Algorithm**: Gradient Boosting Classifier
- Real-time prediction capability
- Handles temporal patterns
- Provides risk stratification

## Project Structure

```
09-sepsis-detection/
├── sepsis_model.py               # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── models/
    ├── sepsis_model.pkl          # Trained model
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

- **ROC-AUC Score**: 0.85-0.90
- **Precision**: 0.75-0.80
- **Recall**: 0.80-0.85
- **F1-Score**: 0.77-0.82

## Top Sepsis Indicators

1. Temperature change (>1°C)
2. Heart rate change (>20 bpm)
3. Respiratory rate change (>5)
4. Lactate level (>2 mmol/L)
5. Procalcitonin (>0.5 ng/mL)
6. WBC abnormality (<4 or >12)
7. Platelet count (<100)
8. pH (<7.3)
9. Age (>65)
10. Immunocompromised status

## Risk Stratification

- **Low Risk** (< 0.3): Continue monitoring
- **Medium Risk** (0.3-0.7): Alert clinicians, prepare interventions
- **High Risk** (> 0.7): Immediate sepsis protocol activation

## Clinical Action Protocol

### For High-Risk Patients
- Activate sepsis alert
- Blood cultures
- Broad-spectrum antibiotics
- Fluid resuscitation
- Lactate monitoring
- ICU consultation

### For Medium-Risk Patients
- Close monitoring
- Repeat labs in 2-4 hours
- Prepare for escalation
- Notify team

### For Low-Risk Patients
- Continue standard care
- Regular monitoring

## Real-World Applications

- **ICU**: Real-time sepsis monitoring
- **Emergency Department**: Triage and early identification
- **Hospitals**: Quality improvement programs
- **Telemedicine**: Remote patient monitoring
- **Wearables**: Continuous health monitoring

## Benefits

1. **Early detection**: 6-hour advance warning
2. **Improved outcomes**: Higher survival rates
3. **Reduced mortality**: Timely intervention
4. **Cost savings**: Shorter ICU stays
5. **Quality metrics**: Sepsis bundle compliance

## Clinical Validation

- Model should be validated on external datasets
- Requires clinical trial before deployment
- Must integrate with clinical workflows
- Regular performance monitoring needed
- Physician override capability essential

## Future Enhancements

1. Multi-step ahead prediction
2. Organ dysfunction prediction
3. Personalized treatment recommendations
4. Integration with EHR systems
5. Real-time alert system

## References

- Seymour, C. W., et al. (2016). Assessment of Clinical Criteria for Sepsis
- Singer, M., et al. (2016). The Third International Consensus Definitions for Sepsis
- Rhee, C., et al. (2017). Incidence and Trends of Sepsis in US Hospitals

## Author

Created by Buse Bircan as part of ML Portfolio Projects

## License

MIT License
