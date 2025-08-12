# Random Forest from Scratch

A clean, educational implementation of the Random Forest algorithm built from the ground up using only NumPy. This project demonstrates core machine learning concepts including decision trees, bootstrap sampling, and ensemble methods.

## 🎯 Overview

This implementation includes:
- **Decision Tree**: A complete binary decision tree with entropy-based splitting
- **Random Forest**: An ensemble of decision trees using bootstrap sampling and feature randomness
- **Breast Cancer Classification**: Practical demonstration on the Wisconsin Breast Cancer dataset

## 🚀 Features

- ✅ Pure NumPy implementation (no external ML libraries for core algorithm)
- ✅ Configurable hyperparameters (tree depth, sample splitting, feature selection)
- ✅ Information gain optimization using entropy
- ✅ Bootstrap sampling for ensemble diversity
- ✅ Comprehensive evaluation with accuracy metrics and confusion matrix
- ✅ Clean, readable code with proper OOP structure

## 📁 Project Structure

```
random_forest_from_scratch/
├── DecisionTree.py          # Core decision tree implementation
├── random_forest.ipynb      # Random forest class and demonstration
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns
├── LICENSE                 # MIT License
└── README.md               # This file
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/hghaemi/random_forest_from_scratch.git
cd random_forest_from_scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebook:
```bash
jupyter notebook random_forest.ipynb
```

## 💻 Usage

### Basic Usage

```python
from DecisionTree import DecisionTree
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load and prepare data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=56
)

# Create and train Random Forest
rf = RandomForest(n_trees=100, max_depth=10, min_sample_split=2)
rf.fit(X_train, y_train)

# Make predictions
predictions = rf.predict(X_test)
```

### Hyperparameter Configuration

```python
# Customize the Random Forest
rf = RandomForest(
    n_trees=50,           # Number of trees in the forest
    max_depth=15,         # Maximum depth of each tree
    min_sample_split=5,   # Minimum samples required to split a node
    n_feature=10          # Number of features to consider at each split
)
```

## 🧮 Algorithm Details

### Decision Tree
- **Splitting Criterion**: Information Gain using entropy
- **Feature Selection**: Random subset of features at each split
- **Stopping Conditions**: Maximum depth, minimum samples, or pure node
- **Prediction**: Traverses tree based on feature thresholds

### Random Forest
- **Bootstrap Sampling**: Each tree trained on random sample with replacement
- **Feature Randomness**: Subset of features considered at each split
- **Ensemble Prediction**: Majority vote across all trees
- **Diversity**: Combines bootstrap sampling and feature randomness

## 📊 Performance

On the Wisconsin Breast Cancer dataset:
- **Dataset**: 569 samples, 30 features
- **Train/Test Split**: 80/20
- **Typical Accuracy**: ~92-95%
- **Classes**: Malignant (0) vs Benign (1)

## 🔧 Class Parameters

### DecisionTree
- `min_samples_split`: Minimum samples required to split (default: 2)
- `max_depth`: Maximum tree depth (default: 100)
- `n_features`: Number of features to consider per split (default: all)

### RandomForest
- `n_trees`: Number of trees in ensemble (default: 100)
- `max_depth`: Maximum depth per tree (default: 10)
- `min_sample_split`: Minimum samples to split (default: 2)
- `n_feature`: Features per split (default: sqrt of total features)

## 🎓 Educational Value

This implementation demonstrates:
1. **Entropy and Information Gain**: How decision trees choose optimal splits
2. **Bootstrap Sampling**: Creating diverse training sets for ensemble methods
3. **Feature Randomness**: Reducing correlation between ensemble members
4. **Ensemble Learning**: Combining weak learners into strong predictors
5. **Object-Oriented Design**: Clean separation of concerns and reusable components

## 🔍 Key Code Concepts

### Information Gain Calculation
```python
def _information_gain(self, y, X_column, threshold):
    parent_entropy = self._entropy(y)
    left_idxs, right_idxs = self._split(X_column, threshold)
    
    # Calculate weighted child entropy
    n = len(y)
    n_l, n_r = len(left_idxs), len(right_idxs)
    e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
    child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
    
    return parent_entropy - child_entropy
```

### Bootstrap Sampling
```python
def _bootstrap_samples(self, X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Feel free to contribute by:
- Adding support for regression tasks
- Implementing additional splitting criteria (Gini impurity)
- Adding cross