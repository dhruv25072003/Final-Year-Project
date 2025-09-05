
# 🧠 ANN Regression Model for Q Factor Prediction

Predicting the **Q factor** of a system using Artificial Neural Networks in Python.

## 📚 Project Overview

This repository contains a clean, beginner-friendly notebook that demonstrates how to build and train an Artificial Neural Network (ANN) for regression.  
**Goal:** Predict the `Q factor` from physical parameters such as:
- `radius`
- `frequency`
- `imaginary frequency`

The workflow includes data preprocessing, feature engineering, normalization, model building using Keras/TensorFlow, training, evaluation, and result visualization.

---

## 🚀 Key Features

- **Comprehensive Data Pipeline:**  
  - Data loading, NaN/outlier handling, and feature scaling.
  - Feature engineering with polynomial terms for increased expressiveness.
- **Custom ANN Architecture:**  
  - Multi-layer architecture with BatchNorm and Dropout for regularization.
- **Robust Training/Evaluation:**  
  - Train/test split, convergence visualizations, and key regression metrics.
- **Easy Visualization:**  
  - Plots of training vs. validation loss, and true vs. predicted outputs.
- **Well-commented for Clarity:**  
  - Every step is documented to ease understanding for newcomers.

---

## 📈 Model Workflow

1. **Data Preparation**
   - Reads input from Excel file.
   - Cleans columns and drops unused or NaN fields.
   - Creates polynomial feature expansions for radius/frequency.

2. **Feature Scaling**
   - Scales features using `RobustScaler` to minimize outlier distortions.

3. **Model Architecture**
   - 64 → 128 → 64 neuron Dense layers, with `ReLU`, `BatchNormalization`, and `Dropout`.
   - Single linear output for regression.

4. **Model Training & Validation**
   - 100 epochs with early stopping potential (edit for more).
   - Track training and validation MSE loss.

5. **Evaluation**
   - Predicts Q on test set.
   - Calculates MSE, MAE, and R² for regression quality.
   - Generates scatter plots for result analysis.

---

## 💻 Usage

### 1. **Clone the Repo**
```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. **Install Requirements**
```
pip install -r requirements.txt
```
(if using a requirements file; otherwise, ensure you have `pandas`, `numpy`, `matplotlib`, `scikit-learn`, and `tensorflow`)

### 3. **Run the Notebook**
Open `Related.ipynb` in Jupyter Notebook or [VS Code](https://code.visualstudio.com/).

---

## 🔬 Example Results

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-image-link/train_val_loss.png" width="400" alt="Training vs Validation Loss" />
  <img src="https://user-images.githubusercontent.com/your-image-link/scatter_true_vs_pred.png" width="400" alt="True vs Predicted Q Scatter" />
</p>
<details>
<summary>Sample Output</summary>

| Q True      | Q Pred    |
|-------------|-----------|
| 528.67      | 564.70    |
| 1055.38     | 591.24    |
| -395.22     | 73.83     |
| ...         | ...       |

</details>

---

## ⚙️ File Structure

```
│   Related.ipynb           # Main notebook with full experiment
│   exML1(1).xlsx           # Input data file (not uploaded for privacy)
│   requirements.txt        # (Optional) Python dependencies
│
└── README.md               # This file
```

---

## 📊 Performance Metrics

- **Mean Squared Error (MSE):** _reported by notebook_
- **Mean Absolute Error (MAE):** _reported by notebook_
- **R² Score:** _reported by notebook_

> **Note:** Performance depends on data distribution, input range, and ANN hyperparameters. Feel free to experiment!

---

## 🤝 Contributions

This repo serves as a template and study reference.  
Feel free to open Issues/PRs to suggest improvements or add more advanced models.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Happy coding and experimenting!**

---

> _If you find this useful, kindly star 🌟 the repo!_
```

***
