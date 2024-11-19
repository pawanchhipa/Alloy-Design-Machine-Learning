import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

class AlloyPredictor:
    def __init__(self):
        self.model = None
    
    def load_alloy_data(self, file_path):
        """
        Loads alloy dataset from a CSV file.
        
        Args:
            file_path (str): Path to the CSV file containing alloy data
            
        Returns:
            tuple: (X, y) where X is features dataframe and y is target series
        """
        data = pd.read_csv(file_path)
        
        # Extracting features (concentration columns) and target (hardness)
        Xcols = data.columns[data.columns.str.contains("C.*")]
        X = data[Xcols]
        y = data['HV']
        
        return X, y

    def perform_kfold_analysis(self, X, y, n_splits=5):
        """
        Performs k-fold cross-validation analysis and returns detailed metrics.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of folds for cross-validation
        
        Returns:
            dict: Dictionary containing lists of metrics for each fold
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        rmse_train_list = []
        rmse_test_list = []
        r2_train_list = []
        r2_test_list = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_test_fold = y.iloc[test_idx]
            
            model = LinearRegression()
            model.fit(X_train_fold, y_train_fold)
            
            y_train_pred = model.predict(X_train_fold)
            y_test_pred = model.predict(X_test_fold)
            
            rmse_train = np.sqrt(mean_squared_error(y_train_fold, y_train_pred))
            rmse_test = np.sqrt(mean_squared_error(y_test_fold, y_test_pred))
            r2_train = r2_score(y_train_fold, y_train_pred)
            r2_test = r2_score(y_test_fold, y_test_pred)
            
            rmse_train_list.append(rmse_train)
            rmse_test_list.append(rmse_test)
            r2_train_list.append(r2_train)
            r2_test_list.append(r2_test)
            
            fold_results.append({
                'fold': fold,
                'rmse_train': rmse_train,
                'rmse_test': rmse_test,
                'r2_train': r2_train,
                'r2_test': r2_test
            })
        
        return {
            'rmse_train': rmse_train_list,
            'rmse_test': rmse_test_list,
            'r2_train': r2_train_list,
            'r2_test': r2_test_list,
            'fold_results': fold_results
        }

    def analytical_solution(self, X, y):
        """
        Compute the analytical solution for linear regression using the normal equation.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            ndarray: Optimal coefficients
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return theta_best

    def gradient_descent(self, X, y, learning_rate=0.01, n_iterations=1000):
        """
        Implement gradient descent for linear regression.
        
        Args:
            X: Feature matrix
            y: Target vector
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations
            
        Returns:
            ndarray: Optimal coefficients
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        theta = np.random.randn(n)
        
        cost_history = []
        
        for iteration in range(n_iterations):
            gradients = (2/m) * X_b.T @ (X_b @ theta - y)
            theta -= learning_rate * gradients
            
            predictions = X_b @ theta
            cost = np.mean((predictions - y) ** 2)
            cost_history.append(cost)
        
        return theta

    def plot_convergence(self, X, y, learning_rate=0.01, n_iterations=1000):
        """
        Plot the convergence of gradient descent.
        
        Args:
            X: Feature matrix
            y: Target vector
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        theta = np.random.randn(n)
        
        cost_history = []
        
        for iteration in range(n_iterations):
            gradients = (2/m) * X_b.T @ (X_b @ theta - y)
            theta -= learning_rate * gradients
            
            predictions = X_b @ theta
            cost = np.mean((predictions - y) ** 2)
            cost_history.append(cost)
        
        plt.figure(figsize=(8, 5))
        plt.plot(cost_history)
        plt.title('Gradient Descent Convergence')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()
