# Credit Card Fraud Detection: Application of ML and cybersecurity in FinTech 

A machine learning project demonstrating fraud detection with realistic performance metrics, proper handling of class imbalance, and critical analysis of model limitations. This project showcases end-to-end ML workflow including data preprocessing, model training, evaluation, and business impact analysis.

## 🎯 Project Overview

This project explores credit card fraud detection using the publicly available Kaggle dataset. Rather than pursuing unrealistically perfect metrics, this implementation focuses on:
- Realistic model performance with proper validation
- Understanding precision-recall tradeoffs in imbalanced classification
- Business impact analysis and production considerations
- Critical evaluation of model limitations

**Key Learning:** Fraud detection is not about achieving 99% accuracy—it's about balancing false positives (angry customers) against false negatives (missed fraud) based on business costs.

## 📁 Dataset

**Source:** [Credit Card Fraud Detection Dataset](https://www.kaggle.com/code/gauravduttakiit/creditcard-fraud-detection-by-logistic-regression/input?select=creditcard.csv) from Kaggle

**Description:** 
- Transactions made by European cardholders in September 2013
- 284,807 transactions with only 492 frauds (0.172% - highly imbalanced)
- Features V1-V28 are PCA-transformed for confidentiality
- Additional features: Time (seconds since first transaction), Amount (transaction value)
- Binary classification: Class 0 (legitimate), Class 1 (fraud)

**Setup Instructions:**
1. Download the dataset from [Kaggle](https://www.kaggle.com/code/gauravduttakiit/creditcard-fraud-detection-by-logistic-regression/input?select=creditcard.csv)
2. Place `creditcard.csv` in the project root directory
3. Run the Jupyter notebook

**Note on Dataset Modification:**
To demonstrate realistic fraud detection challenges, I used a 30% sample and removed some highly predictive PCA features (V15-V19). The original dataset's PCA features are so well-separated that models easily achieve >99% ROC-AUC, which doesn't reflect real-world complexity where fraud patterns are constantly evolving.

## 🔬 Methodology

### The Class Imbalance Problem
Fraud detection faces extreme class imbalance (~0.2% fraud rate). Traditional approaches fail because:
- Models learn to predict everything as "legitimate" and still get 99.8% accuracy
- Standard train/test splits result in tiny fraud samples in test sets
- Evaluation metrics like accuracy become misleading

### Solutions Implemented

**1. SMOTE (Synthetic Minority Over-sampling Technique)**
- Applied only to training data to avoid data leakage
- Increased fraud representation to 50% of legitimate transactions
- Creates synthetic fraud examples by interpolating between existing cases

**2. Adjusted Classification Threshold**
- Lowered from default 0.5 to 0.3
- Increases fraud detection sensitivity
- Tradeoff: More false alarms, but catches more fraud

**3. Class-Weighted Models**
- Added `class_weight='balanced'` to penalize fraud misclassification
- Forces model to pay attention to minority class

**4. Proper Evaluation Metrics**
- Focus on Recall (fraud catch rate) and F1-Score (balance)
- ROC-AUC for overall discrimination ability
- Accuracy is reported but not prioritized

## ⚙️ Models Evaluated

### 1. Logistic Regression (Baseline)
Simple linear model to establish baseline performance. Results show why linear models struggle with complex fraud patterns.

### 2. Random Forest (Production Model)
Non-linear ensemble model that captures complex interaction patterns between features.

## 📊 Results Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Logistic Regression** | 97.1% | 4.6% | 88.2% | 8.7% | 94.9% |
| **Random Forest** | 99.9% | 51.8% | 85.3% | 64.4% | 97.4% |

### Model Interpretation

**Logistic Regression Performance:**
- Catches 88% of fraud (good recall) but with only 4.6% precision
- This means 95.4% of flagged transactions are FALSE ALARMS
- Would require manual review of thousands of legitimate transactions
- **Business Impact:** Unacceptable customer friction, high operational costs
- **Verdict:** Not suitable for production use

**Random Forest Performance:**
- Catches 85.3% of fraud with 51.8% precision
- About half of flagged transactions are actual fraud
- Misses 5 out of 34 fraud cases in test set (15% miss rate)
- **Business Impact:** Reasonable balance between fraud prevention and customer experience
- **Verdict:** Strong candidate for production deployment

### Confusion Matrix Breakdown (Random Forest)

<p align="center">
<b>Confusion Matrix Results</b><br/>
<img src="https://i.imgur.com/fxa4vNf.png" height="80%" width="80%" alt="Confusion Matrix"/>
</p>

**Key Metrics:**
- **True Negatives:** 21,300 legitimate transactions correctly approved
- **False Positives:** 27 legitimate transactions incorrectly flagged (customer friction)
- **False Negatives:** 5 fraudulent transactions missed (financial loss)
- **True Positives:** 29 fraudulent transactions correctly caught (fraud prevented)

## 💰 Business Impact Analysis

Assuming:
- Average fraud amount: $100
- Manual review cost: $5 per flagged transaction
- Customer churn risk: 10% of falsely declined customers

**Random Forest Financial Impact:**
- Fraud Prevented: $2,900 (29 cases × $100)
- Fraud Missed: $500 (5 cases × $100)
- Manual Review Cost: $135 (27 false alarms × $5)
- Potential Customer Loss: ~3 customers

**Net Benefit:** $2,265 saved  
**ROI:** $16.78 saved for every $1 spent on manual review

**Why This Matters:** In production, you'd optimize threshold based on your specific business costs. Banks typically tolerate higher false positive rates because fraud losses exceed customer service costs.

## 📉 Visualizations

<p align="center">
<b>Class Distribution Plot- Shows extreme imbalance between classes (log scale)</b><br/>
<img src="https://i.imgur.com/X9GiSL9.png" height="80%" width="80%" alt="Pipeline Overview"/>
<br /><br />
<b>ROC Curve- Model discrimination ability comparison</b><br/>
<img src="https://i.imgur.com/u7a2zqq.png" height="80%" width="80%" alt="Confusion Matrix"/>
<b>Threshold Analysis- Precision-recall tradeoff across different thresholds</b><br/>
<img src="https://i.imgur.com/amRJYWj.png" height="80%" width="80%" alt="Confusion Matrix"/>
</p>


## ⚠️ Project Limitations & Real-World Considerations

**1. Statistical Validity**
- Only 34 fraud cases in test set (small sample for robust conclusions)
- Would need 500+ fraud cases for production-level confidence
- Single train/test split without cross-validation

**2. SMOTE Concerns**
- Synthetic fraud may not match real fraud patterns
- Risk of overfitting to interpolated examples
- Real fraudsters are creative—synthetic patterns are mathematical

**3. Temporal Validation Missing**
- No time-based split (train on older data, test on recent data)
- Fraud tactics evolve constantly (concept drift)
- Model trained on January fraud may miss February's new schemes

**4. PCA Feature Limitations**
- Cannot interpret what features actually mean
- Limits domain knowledge application
- Can't explain predictions to customers or regulators

**5. Evaluation Bias**
- Test set from same time period as training
- Real-world: future fraud looks different than past fraud
- Performance likely overestimated

### Production Deployment Requirements

**Before this model goes live, you'd need:**

1. **Time-Based Validation**
   - Train on months 1-6, validate on month 7
   - Test on month 8 to measure real future performance

2. **Larger Validation Dataset**
   - Minimum 500-1000 fraud cases for statistical power
   - Stratified by fraud type, amount, merchant category

3. **A/B Testing Framework**
   - Shadow mode: Compare model decisions to current system
   - Gradual rollout: Start with 10% of traffic
   - Measure real business metrics: fraud caught, customer complaints

4. **Monitoring System**
   - Track model performance over time (concept drift detection)
   - Alert when precision/recall degrade
   - Automatic model retraining pipeline

5. **Explainability Layer**
   - SHAP values for regulatory compliance
   - Customer service needs explanations: "Why was I declined?"
   - Audit trail for investigations

6. **Cost-Sensitive Learning**
   - Incorporate actual business costs into training
   - Different fraud amounts should have different weights
   - Consider customer lifetime value in false positive costs

7. **Integration with Review Queue**
   - Route borderline cases (0.3 < probability < 0.7) to human review
   - High-confidence predictions can be auto-approved/declined
   - Feedback loop: Reviewers label cases for model retraining

### Model Improvement Strategies

**Short-term:**
- Try XGBoost/LightGBM for better performance
- Implement cross-validation for robust estimates
- Add temporal features (hour of day, day of week)

**Medium-term:**
- Online learning to adapt to new fraud patterns
- Ensemble multiple models for stability
- Feature engineering from Amount and Time

**Long-term:**
- Graph neural networks (transaction network analysis)
- Deep learning on raw transaction sequences
- Multi-modal models (transaction + user behavior)

## 🧪 Failed Dataset Experiment

I also tested a synthetic transaction dataset that **completely failed** to produce a working model. This experiment demonstrates an important lesson in ML:

**Key Finding:** No amount of hyperparameter tuning can fix a fundamentally flawed dataset.

**Results:**
- ROC-AUC stuck at ~0.52 (essentially random guessing)
- Removing ID columns didn't help
- High recall (72%) but terrible precision (5%)
- Model was just randomly guessing "fraud" aggressively

**Root Cause:** The dataset had randomly assigned fraud labels with no underlying patterns. The features contained no meaningful signals that correlate with fraud.

**Lesson Learned:** Machine learning models can only learn patterns that actually exist in the data. Garbage in = garbage out, no matter how sophisticated your model is.

This failed experiment is included in the repository as `FraudDetection.ipynb` as a teaching example.

## 🖼️ Sample Visualizations

<p align="center">
<b>ROC-AUC Comparison: Failed vs Working Dataset</b><br/>
<img src="https://i.imgur.com/fG6OvxX.png" height="40%" width="40%" alt="Failed Dataset ROC-AUC"/>
<img src="https://i.imgur.com/AhC340H.png" height="40%" width="40%" alt="Credit Card Dataset ROC-AUC"/>
<br /><br />
<b>ML Pipeline & Data Flow</b><br/>
<img src="https://i.imgur.com/3Ju9PsI.png" height="80%" width="80%" alt="Pipeline Overview"/>
<br /><br />
<b>Confusion Matrix Results</b><br/>
<img src="https://i.imgur.com/BRuG53Y.png" height="80%" width="80%" alt="Confusion Matrix"/>
</p>

## 🧰 Tech Stack

- **Python 3.8+**
- **pandas** - Data manipulation
- **numpy** - Numerical operations
- **scikit-learn** - ML models and preprocessing
- **imbalanced-learn** - SMOTE implementation
- **matplotlib** - Visualizations
- **seaborn** - Statistical plots

## 📦 Project Structure

```
credit-card-fraud-detection/
├── README.md
├── fraud_detection.ipynb          # Negative example
├── fraud_detection.csv
├── creditcard-2.ipynb           # Dataset (download separately)            
```

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/marriammahmed/AI_MLFraudDetectionFinTechAlgo.git
cd credit-card-fraud-detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Visit [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/code/gauravduttakiit/creditcard-fraud-detection-by-logistic-regression/input?select=creditcard.csv)
- Download `creditcard.csv`
- Place in project root directory

4. **Run the notebook**
```bash
jupyter notebook creditcard-2.ipynb
```

## 🎓 Key Takeaways

1. **Accuracy is misleading for imbalanced data** - A model predicting all "legitimate" gets 99.8% accuracy but is useless

2. **Precision vs Recall is a business decision** - Banks can tolerate false positives, but missing fraud is costly

3. **SMOTE helps but has limitations** - Synthetic data doesn't capture real fraud creativity

4. **Perfect metrics are suspicious** - Real-world fraud detection is hard; 85% recall is realistic and valuable

5. **Domain knowledge matters** - Understanding the business problem guides model design and threshold selection

6. **No amount of ML can fix bad data** - As shown in the failed experiment, patterns must exist to be learned

## 📚 References & Further Reading

- Original Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/code/gauravduttakiit/creditcard-fraud-detection-by-logistic-regression/input?select=creditcard.csv)
- SMOTE Paper: Chawla et al. (2002) "SMOTE: Synthetic Minority Over-sampling Technique"
- Imbalanced Classification Guide: [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
- Cost-Sensitive Learning: Elkan (2001) "The Foundations of Cost-Sensitive Learning"



**Note:** This is an educational project using publicly available data. Real production fraud detection systems require significantly more sophisticated approaches, larger datasets, continuous monitoring, and integration with business operations. AI tools where used during the course of this project for time efficiency
