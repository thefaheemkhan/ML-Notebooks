### **A Complete Guide to Logistic Regression: Theory, Formulas, and Metrics**

Logistic Regression is one of the most fundamental and widely used algorithms for binary classification in machine learning. It is simple to implement, computationally efficient, and interpretable, making it a favorite choice for both beginners and seasoned practitioners.

This comprehensive guide will serve as your one-stop resource to understand everything about Logistic Regression—from its mathematical foundation to its evaluation metrics.

---

## **What Is Logistic Regression?**

Logistic Regression is a statistical model used to predict the probability of a binary outcome (e.g., success/failure, 0/1, true/false). While it shares similarities with linear regression, it uses a non-linear transformation to output probabilities rather than continuous values.

### **Key Characteristics:**
- It models the relationship between the input features and the log-odds of the target variable.
- The output lies between 0 and 1, interpreted as a probability.
- It is widely used in fields like finance (credit scoring), healthcare (disease prediction), and marketing (churn prediction).

---

## **1. Hypothesis of Logistic Regression**

Logistic Regression uses the **sigmoid function**, also known as the logistic function, to model probabilities. The sigmoid function transforms any real-valued number into a range between 0 and 1.

\[
h_\theta(x) = \frac{1}{1 + e^{-\theta^T x}}
\]

Where:
- \( h_\theta(x) \): Predicted probability that the target \( y \) is 1 given the input \( x \).
- \( \theta \): Vector of model parameters (weights and bias).
- \( x \): Vector of input features.

For binary classification, the decision rule is:
\[
\hat{y} = 
\begin{cases} 
1 & \text{if } h_\theta(x) \geq 0.5 \\ 
0 & \text{if } h_\theta(x) < 0.5
\end{cases}
\]

---

## **2. The Cost Function**

To train the model, we minimize the **Log-Loss**, also called the **Cross-Entropy Loss**, which measures the error between the predicted probabilities and the true labels.

### **Formula:**
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
\]

Where:
- \( m \): Number of training examples.
- \( y^{(i)} \): Actual label (\( 0 \) or \( 1 \)) for the \( i \)-th training example.
- \( h_\theta(x^{(i)}) \): Predicted probability for the \( i \)-th training example.

### Why Not Use Mean Squared Error (MSE)?
MSE is unsuitable for Logistic Regression because:
1. It results in a non-convex cost function, making optimization difficult.
2. Probabilities can exceed the range [0,1], which is mathematically incorrect for classification tasks.

---

## **3. Optimization: Gradient Descent**

Logistic Regression uses **Gradient Descent** to minimize the cost function and optimize the parameters \( \theta \).

### **Gradient of the Cost Function**
The partial derivative of the cost function with respect to \( \theta_j \) is:
\[
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\]

This tells us the direction and magnitude of change required to minimize the cost.

### **Gradient Descent Update Rules**
Gradient Descent updates the parameters iteratively:
\[
\theta_j := \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}
\]

Where:
- \( \alpha \): Learning rate (step size).
- \( \frac{\partial J(\theta)}{\partial \theta_j} \): Gradient of the cost function with respect to \( \theta_j \).

### **Gradient Descent Variants**

#### **1. Batch Gradient Descent (BGD)**
Uses the entire dataset to compute gradients at each iteration.

**Normal Formula:**
\[
\theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\]

**Vectorized Formula:**
\[
\theta := \theta - \alpha \frac{1}{m} X^T (h_\theta(X) - y)
\]

#### **2. Stochastic Gradient Descent (SGD)**
Uses a single training example to update the parameters, which introduces randomness but speeds up computation.

**Normal Formula:**
\[
\theta_j := \theta_j - \alpha \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\]

**Vectorized Formula (for one example):**
\[
\theta := \theta - \alpha x^{(i)} \cdot \left( h_\theta(x^{(i)}) - y^{(i)} \right)
\]

#### **3. Mini-Batch Gradient Descent**
Uses a small subset (mini-batch) of the dataset to compute gradients, balancing computation time and convergence stability.

**Normal Formula:**
\[
\theta_j := \theta_j - \alpha \frac{1}{b} \sum_{i \in \mathcal{B}} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)}
\]

**Vectorized Formula:**
\[
\theta := \theta - \alpha \frac{1}{b} X_\mathcal{B}^T (h_\theta(X_\mathcal{B}) - y_\mathcal{B})
\]

---

## **4. Regularization**

Regularization prevents overfitting by penalizing large parameter values.

### **L2 Regularization (Ridge)**
Adds \( \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2 \) to the cost function:
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2
\]

### **L1 Regularization (Lasso)**
Adds \( \frac{\lambda}{m} \sum_{j=1}^n |\theta_j| \) to the cost function:
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \frac{\lambda}{m} \sum_{j=1}^n |\theta_j|
\]

### **Elastic Net Regularization**
Combines L1 and L2 penalties:
\[
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] + \lambda_1 \sum_{j=1}^n |\theta_j| + \frac{\lambda_2}{2m} \sum_{j=1}^n \theta_j^2
\]

---

## **5. Evaluation Metrics**

Logistic Regression can be evaluated using the following metrics:

1. **Accuracy:**
   \[
   \text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
   \]

2. **Precision:**
   \[
   \text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP) + False Positives (FP)}}
   \]

3. **Recall (Sensitivity):**
   \[
   \text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP) + False Negatives (FN)}}
   \]

4. **F1 Score:**
   \[
   \text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
   \]

5. **Log-Loss (Cross-Entropy Loss):**
   \[
   \text{Log-Loss} = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]
   \]

6. **AUC-ROC:**
   - **

ROC Curve:** Plots True Positive Rate (TPR) vs. False Positive Rate (FPR).
   - **AUC:** Measures the area under the ROC curve to evaluate the model’s ability to distinguish between classes.

---

## **When to Use Logistic Regression?**

- The target variable is binary (e.g., spam/not spam).
- Features and log-odds have a linear relationship.
- Simplicity and interpretability are crucial.

---

This guide combines theory, formulas, and best practices to provide a deep understanding of Logistic Regression. Use it as your go-to reference!
