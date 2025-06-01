# Projector & Predictor

**Projector & Predictor** is a Python-based machine learning project that applies **Principal Component Analysis (PCA)** and **Linear Regression** from scratch using NumPy.  
It explores dimensionality reduction and predictive modeling through eigen decomposition, polynomial fitting, and MSE evaluation — without using external ML libraries.

---

## Tech Stack
**Language:** Python 3  
**Libraries:** NumPy, Matplotlib  
**Techniques:** PCA, Covariance Matrix, Eigendecomposition, Linear Regression, Polynomial Expansion, Mean Squared Error  
**Tools:** Command-line interface, Auto-grader, Data Visualization

---

## Question 1: Dimensionality Reduction

This section implements **Principal Component Analysis (PCA)** from scratch to reduce high-dimensional data into 2D or 3D spaces while preserving as much variance as possible. It includes visualizations and `.npz` output files for reproducibility.

### Supported Datasets
- `Swiss_Roll` – Non-linear 3D manifold (good for visual demo)
- `toy_data` – 2D structured dataset
- `MNIST` – Subset of digit "3" images
- `simple_data` – Minimal dataset for auto-grading

### How PCA Works
- Computes the mean vector and centers the data
- Calculates the covariance matrix
- Performs eigendecomposition to extract principal components
- Projects data onto top `k` eigenvectors to reduce dimensions

### Run PCA

```bash
# Swiss Roll 3D → 2D
python3 DR.py --data Swiss_Roll --method PCA --out_dim 2 --display --save

# Toy Data 2D → 2D (demo)
python3 DR.py --data toy_data --method PCA --out_dim 2 --display

# MNIST 784D → 5D + reconstruction
python3 DR.py --data MNIST --method PCA --out_dim 5 --display --save

# Run autograder
python3 DR.py --method PCA --auto_grade

```

---

## Linear Regression

This section implements **Linear Regression** from scratch using the **normal equation** with optional **polynomial feature expansion**. The goal is to fit a model to various synthetic datasets and evaluate performance using Mean Squared Error (MSE).

### How Linear Regression Works
- Transforms inputs using polynomial powers up to a specified degree
- Solves for weights using the normal equation:
  \[
  \mathbf{w} = (X^T X)^{-1} X^T y
  \]
- Predicts output values and calculates MSE on training and test data
- Optionally displays plots and saves `.npz` result files

### Run Linear Regression

```bash
# Fit linear model (degree 1)
python3 LR.py --data linear --polynomial 1 --display --save

# Fit quadratic model (degree 2)
python3 LR.py --data quadratic --polynomial 2 --display --save

# Auto-grader mode
python3 LR.py --data simple --auto_grade

```

---

## What I Learned

### Principal Component Analysis (PCA)
- Understood how PCA reduces high-dimensional data by maximizing variance preservation.
- Learned to compute the **mean**, **covariance matrix**, and perform **eigendecomposition** from scratch.
- Visualized how PCA works well on linearly separable data (e.g., toy_data), but fails on non-linear manifolds like Swiss Roll.
- Applied PCA to real-world MNIST data and analyzed reconstructed images based on top components.

### Linear Regression
- Implemented linear regression using the **normal equation**, without libraries like `scikit-learn`.
- Applied **polynomial feature expansion** to capture non-linear relationships.
- Evaluated model performance using **Mean Squared Error (MSE)** on both training and test sets.
- Interpreted and visualized model behavior for different polynomial degrees to understand underfitting vs. overfitting.

This project strengthened my understanding of both **unsupervised** and **supervised** learning foundations — and gave me confidence in building ML pipelines from scratch.

---

## 📂 Project Structure

```bash
Projector-and-Predictor/
├── DR.py # Implements PCA and Laplacian Eigenmaps
├── LR.py # Implements linear and polynomial regression
├── Swiss_Roll_2.png # PCA result on Swiss Roll dataset
├── toy_data_2.png # PCA result on toy dataset
├── MNIST_2.png # PCA reconstruction result on MNIST
├── linear_1.png # Linear regression fit (degree 1)
├── quadratic_2.png # Quadratic regression fit (degree 2)
├── Results_*.npz # Saved NumPy results (weights, PCA matrices)
├── collaboration.txt # Notes on project collaboration
└── README.md # Project documentation (this file)
```
---
## Conclusion

This project was a rewarding deep dive into two foundational machine learning techniques — **Principal Component Analysis** and **Linear Regression** — implemented entirely from scratch.

I gained hands-on experience in:
- Unsupervised learning and data transformation through PCA
- Supervised learning with interpretable linear models
- Visualizing model behavior and identifying performance patterns (e.g., underfitting and overfitting)
- Working without high-level ML libraries, reinforcing mathematical understanding

> From reducing dimensions to predicting outcomes, this project strengthened my grasp of machine learning fundamentals — and the confidence to build models end-to-end with just Python and NumPy.

