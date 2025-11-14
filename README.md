# Cybersecurity in FinTech: Credit Card Fraud Detection Machine Learning Model

This project applies basic machine learning models to detect fraudulent credit card transactions. The goal was to understand how data imbalance, feature selection, and model tuning affect fraud detection performance.

## 📁 Dataset
The dataset used is the publicly available **Credit Card Fraud Detection dataset** from Kaggle. It contains transactions made by European cardholders in 2013. Each transaction has anonymized PCA-transformed features (`V1`–`V28`), an `Amount`, and a `Class` label (0 = legitimate, 1 = fraud).

Due to the dataset’s design, most features already separate frauds very clearly, so models tend to achieve unrealistically high ROC-AUC scores (~99%). To simulate more realistic results, I:
- used only 5% of the dataset,  
- removed highly predictive PCA features (`V1–V20`),  
- added noise to make patterns less obvious.  

These adjustments reduced model performance to around **70–80% ROC-AUC**, which is more realistic for actual fraud detection systems.

## ⚙️ Models Used
Two models were tested:
- Logistic Regression (baseline)
- Random Forest (non-linear)

Both models were trained on scaled numeric features after data preprocessing.  

## 📊 Results Summary
| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|--------|-----------|-----------|---------|-----------|----------|
| Logistic Regression | ~98% | ~0.60 | ~0.50 | ~0.55 | ~0.75 |
| Random Forest | ~99% | ~0.70 | ~0.65 | ~0.67 | ~0.78 |

Fraud detection models often show high accuracy because the dataset is extremely imbalanced (frauds are <0.2% of data). The **ROC-AUC** and **recall** are better indicators of performance.

## 📉 Visualization
Three simple plots are included:
1. **Class distribution** – to show imbalance between normal and fraud transactions  
2. **Confusion matrix** – to visualize false positives and false negatives  
3. **ROC curve** – to compare sensitivity and specificity  

## 🧪 Dataset Experiment Note
I also tested another dataset to simulate transactions, but it failed completely — no amount of fine-tuning helped.  
Because the data was randomly generated, there were **no underlying behavioral patterns** for the model to learn. It either **overfit** (memorizing transaction IDs and noise) or performed at **coin-flip accuracy (50%)** when regularized.  
This shows how machine learning only works when real, meaningful patterns exist in the data.

ROC-AUC was STILL ~0.52 (essentially random). This confirms the dataset itself has no real fraud patterns – the features don't contain meaningful signals to distinguish fraud from legitimate transactions.

🔴 The Reality:
The dataset appears to be synthetic/artificial with randomly assigned fraud labels. No amount of tuning will fix this because there's nothing to learn.

**Evidence:**
- ROC-AUC stuck at 0.52 (coin flip)  
- Removing `User_ID` didn't help (it wasn't the only issue)  
- Even with 72% recall, the model is just guessing aggressively  
- Precision is 5% (95% of fraud predictions are wrong)  

## 🖼️ Sample Walk-through

<p align="center">
ROC-AUC Comparison: <br/>
<img src="https://i.imgur.com/fG6OvxX.png" height="40%" width="40%" alt="Failed Dataset ROC-AUC"/>
<img src="https://i.imgur.com/AhC340H.png" height="40%" width="40%" alt="Credit Card Dataset ROC-AUC"/>
<br /><br />
Pipeline & Preprocessing: <br/>
<img src="https://i.imgur.com/3Ju9PsI.png" height="80%" width="80%" alt="Pipeline Overview"/>
<br /><br />
Old Dataset Feature Selection: <br/>
<img src="https://i.imgur.com/uQJiymx.png" height="80%" width="80%" alt="Feature Selection"/>
<br /><br />
Confusion Matrix: <br/>
<img src="https://i.imgur.com/BRuG53Y.png" height="80%" width="80%" alt="Confusion Matrix"/>
</p>

## 🧰 Tech Stack
- Python  
- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

## 📦 Project Components (Files Included)
- **Project Report (PDF)** – full explanation of methodology, results, and analysis  
- **Dataset 1:** Credit Card Fraud Dataset (working dataset)  
- **Dataset 2:** Failed synthetic/random dataset (no learnable pattern)  
- **Notebook 1:** Credit Card Fraud Detection (full pipeline + visualizations)  
- **Notebook 2:** Failed Dataset Experiment (ROC-AUC ~0.5 demonstration)  
