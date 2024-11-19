from alloy_ml_prediction import AlloyPredictor

def main():
    # Initialize predictor
    predictor = AlloyPredictor()
    
    # Load data with the correct relative path
    X, y = predictor.load_alloy_data('dataset/alloy-confp-train-data_v2.csv')
    
    # Perform k-fold analysis
    results = predictor.perform_kfold_analysis(X, y, n_splits=5)
    print("\nK-fold Analysis Results:")
    for fold in results['fold_results']:
        print(f"\nFold {fold['fold']}:")
        print(f"Training RMSE: {fold['rmse_train']:.4f}")
        print(f"Testing RMSE: {fold['rmse_test']:.4f}")
        print(f"Training R²: {fold['r2_train']:.4f}")
        print(f"Testing R²: {fold['r2_test']:.4f}")
    
    # Compare analytical and gradient descent solutions
    analytical_coef = predictor.analytical_solution(X, y)
    gradient_coef = predictor.gradient_descent(X, y)
    
    print("\nCoefficient Comparison:")
    print("Analytical Solution:", analytical_coef)
    print("Gradient Descent:", gradient_coef)
    
    # Plot convergence
    predictor.plot_convergence(X, y)

if __name__ == "__main__":
    main()
