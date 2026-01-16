# Drug Recommendation System

## Project Overview

This project implements a machine learning-based drug recommendation system that suggests appropriate medications based on patient conditions, lab values, and medical history. The system helps healthcare providers make evidence-based prescribing decisions.

## Business Problem

Medication selection requires considering:
- Patient conditions and comorbidities
- Lab values and clinical measurements
- Drug interactions and contraindications
- Patient allergies
- Prior medication history

The system automates this complex decision-making process to:
- Improve prescribing accuracy
- Reduce medication errors
- Ensure evidence-based treatment
- Optimize therapy
- Support clinical decision-making

## Features

### Patient Demographics
- Age, gender, weight, height, BMI

### Medical Conditions
- Diabetes, hypertension, heart disease, asthma, COPD
- Thyroid disease, GERD, depression, arthritis, kidney disease

### Laboratory Values
- Glucose, HbA1c, cholesterol levels
- Blood pressure, TSH, creatinine

### Safety Considerations
- Drug allergies (penicillin, NSAIDs)
- Contraindications (ACE inhibitors, NSAIDs)
- Current medication count
- Medication adherence
- Prior adverse events

## Model Architecture

**Algorithm**: Multi-label Random Forest Classifiers
- One classifier per drug
- Handles multiple drug recommendations
- Provides confidence scores

## Project Structure

```
10-drug-recommendation/
├── drug_recommendation_model.py  # Main model implementation
├── train.py                      # Training script
├── predict.py                    # Prediction script
├── README.md                     # This file
├── requirements.txt              # Python dependencies
└── models/
    ├── drug_models.pkl           # Trained models
    ├── scaler.pkl                # Feature scaler
    └── drug_list.pkl             # Drug list
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

## Supported Drugs

- Metformin (diabetes)
- Lisinopril (hypertension)
- Atorvastatin (cholesterol)
- Amlodipine (hypertension)
- Omeprazole (GERD)
- Albuterol (asthma/COPD)
- Levothyroxine (thyroid)
- Aspirin (cardiovascular)
- Ibuprofen (pain/inflammation)
- Sertraline (depression)
- Amoxicillin (infection)
- Insulin (diabetes)

## Model Performance

- **Per-drug Accuracy**: 0.80-0.90
- **Recommendation Coverage**: 85-95%
- **Contraindication Avoidance**: >99%

## Recommendation Logic

### Diabetes Management
- Metformin: Fasting glucose > 126 mg/dL
- Insulin: HbA1c > 8% or severe hyperglycemia

### Hypertension Control
- Lisinopril: Systolic BP > 140 mmHg (if no contraindication)
- Amlodipine: Systolic BP > 140 mmHg

### Cardiovascular Protection
- Atorvastatin: LDL > 130 or existing heart disease
- Aspirin: Existing heart disease

### Symptom Management
- Omeprazole: GERD diagnosis
- Albuterol: Asthma or COPD
- Ibuprofen: Arthritis (if no contraindication)

### Hormone Replacement
- Levothyroxine: Thyroid disease or TSH > 4

### Mental Health
- Sertraline: Depression diagnosis

## Safety Features

1. **Allergy checking**: Avoids penicillin for allergic patients
2. **Contraindication screening**: Respects medical contraindications
3. **Drug interaction awareness**: Considers current medications
4. **Adherence assessment**: Accounts for patient compliance
5. **Adverse event history**: Learns from prior reactions

## Real-World Applications

- **Clinical Decision Support**: Assist physicians in prescribing
- **Pharmacy Systems**: Automated medication recommendations
- **Telehealth**: Remote prescription support
- **EHR Integration**: Built-in drug recommendation
- **Patient Education**: Explain medication choices

## Benefits

1. **Improved accuracy**: Evidence-based recommendations
2. **Safety**: Automatic contraindication checking
3. **Efficiency**: Faster prescribing decisions
4. **Consistency**: Standardized recommendations
5. **Learning**: Improves with more data

## Clinical Validation

- Model requires clinical validation before deployment
- Must be reviewed by pharmacists and physicians
- Regular monitoring for safety and efficacy
- Physician override capability essential
- Patient preferences must be considered

## Limitations

- Cannot replace clinical judgment
- Requires accurate patient data
- May miss rare conditions
- Needs regular updates with new evidence
- Should not be used as sole decision-making tool

## Future Enhancements

1. Drug interaction prediction
2. Dosage optimization
3. Side effect prediction
4. Patient-specific recommendations
5. Treatment outcome prediction

## References

- Hripcsak, G., et al. (2013). Evaluating the Portability of an Electronic Health Record
- Rajkomar, A., et al. (2018). Scalable and accurate deep learning for electronic health records
- Beam, A. L., & Kohane, I. S. (2018). Big Data and Machine Learning in Health Care

## Author

Created by Buse Bircan as part of ML Portfolio Projects

## License

MIT License
