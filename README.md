Name: Sayantika Chakraborty
Roll No.: ME22B190

# DA5401 - Assignment 2  
**Dimensionality Reduction, Visualization, and Classification Performance** 

## Objective
This assignment applies **Principal Component Analysis (PCA)** to the **Mushroom Dataset** from Kaggle. 
The goals are to:
1. Perform one-hot encoding, standardization. 
2. Apply **PCA** for dimensionality reduction and visualize class separability. 
3. Train and evaluate a **Logistic Regression classifier** on both the original and PCA-transformed data. 
4. Compare performance and analyze the trade-offs between **dimensionality reduction and information loss**. 

## Dataset
- Source: [Kaggle - Mushroom Classification](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- Records: 8124 mushrooms 
- Features: 22 categorical features (cap shape, odor, gill size, etc.) 
- Target: `class` → 
  - `e`: Edible 
  - `p`: Poisonous 

After **one-hot encoding**, the dataset expands to **117 binary features**. 

## Workflow

### Part A: Preprocessing
- Load dataset using `pandas`. 
- Separate target (`y`) from features (`X`). 
- Apply **one-hot encoding** to categorical features. 
- Standardize features using `StandardScaler`. 

### Part B: Principal Component Analysis
- Apply **PCA** without specifying components. 
- Create a **scree plot** (explained variance + cumulative variance). 
- Select number of PCs to retain **≥95% variance** (~59 PCs). 
- Visualize separability: 
  - Scatter plot with **PC1 vs PC2**. 
  - Pair plots of first 4 PCs. 
  - Found that **PC1 vs PC10** gave the best visual separation. 

### Part C: Logistic Regression Performance
- Train Logistic Regression on: 
  - **Original standardized data** (117 features).
  - **PCA-transformed data** (~59 PCs). 
- Evaluate using **accuracy, precision, recall, F1-score**. 
- Compare results: 
  - Both models achieve **near-perfect accuracy** (~100%). 
  - PCA drastically reduces dimensionality while preserving performance. 

## Key Insights
- **One-hot encoding** expands features but introduces **collinearity**. 
- **PCA** creates **uncorrelated principal components**, confirmed by covariance heatmaps. 
- The **condition number** drops significantly after PCA → model becomes more numerically stable. 
- **Logistic Regression** performs equally well on PCA data, proving PCA preserved discriminative information. 
- PCA improves **efficiency, stability, and interpretability**, even if accuracy doesn’t change much. 

## Visualizations
- **Scree plot**: Variance explained by each PC. 
- **PC1 vs PC2 scatter**: Partial separation of edible vs poisonous mushrooms. 
- **PC1 vs PC10 scatter**: Best 2D separation. 
- **Covariance heatmaps**: Original vs PCA space (uncorrelated features). 

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub

