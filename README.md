# League of Legends - Logistic Regression Classifier

This project builds a Logistic Regression model using PyTorch to classify League of Legends match outcomes.  
It covers everything from data loading to model optimization, evaluation, hyperparameter tuning, and feature importance analysis.

---

## 📂 Project Structure

1. **Data Preparation**
   - Load dataset
   - Split into features (X) and labels (y)
   - Train-test split
   - Scale features
   - Convert data to PyTorch tensors

2. **Model Building**
   - Create a custom Logistic Regression model using `torch.nn.Module`
   - Use Sigmoid activation for binary classification

3. **Training**
   - Train the model using Binary Cross Entropy Loss (`nn.BCELoss`) and SGD optimizer
   - Track loss during training

4. **Evaluation**
   - Calculate training and testing accuracy
   - Use thresholds (0.5) to determine class labels

5. **Optimization**
   - Add **L2 Regularization** (Weight Decay) in SGD optimizer
   - Retrain model to prevent overfitting

6. **Visualization**
   - Generate Confusion Matrix
   - Plot ROC Curve and compute AUC
   - Create Classification Report (Precision, Recall, F1-Score)

7. **Model Saving and Loading**
   - Save model weights to Google Drive using `torch.save`
   - Load model weights for future predictions

8. **Hyperparameter Tuning**
   - Experiment with different learning rates: `[0.01, 0.05, 0.1]`
   - Compare test accuracies and select the best learning rate

9. **Feature Importance**
   - Extract weights from the trained model
   - Create a DataFrame to display feature importances
   - Visualize feature importances with a bar plot

---

## 🛠️ Technologies Used

- Python 3
- PyTorch
- Pandas
- Numpy
- Matplotlib
- scikit-learn

---

## 🚀 How to Run the Project

1. Clone the repository or copy the notebook into your environment.
2. Mount Google Drive in Colab (if using Colab):

```python
from google.colab import drive
drive.mount('/content/drive')
