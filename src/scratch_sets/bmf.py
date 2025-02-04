import numpy as np
from tqdm import trange
class BinaryMatrixFactorization:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        """
        Initialize Binary Matrix Factorization.
        
        Parameters:
        -----------
        n_components : int
            Number of components (rank of factorization)
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        
    def _initialize_factors(self, X):
        """Initialize factor matrices W and H randomly."""
        n_samples, n_features = X.shape
        self.W = np.random.randint(0, 2, size=(n_samples, self.n_components))
        self.H = np.random.randint(0, 2, size=(self.n_components, n_features))
    
    def _binary_mult_update(self, X, W, H):
        """Update rule for binary matrix factorization."""
        WH = np.dot(W, H)
        X_diff = X - WH
        
        # Update W
        W_update = np.zeros_like(W)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                # Try flipping each bit and keep if it improves the reconstruction
                W[i, j] = 1 - W[i, j]
                new_error = np.sum((X - np.dot(W, H))**2)
                old_error = np.sum(X_diff**2)
                
                if new_error >= old_error:
                    W[i, j] = 1 - W[i, j]  # Flip back if no improvement
                else:
                    X_diff = X - np.dot(W, H)
        
        # Update H
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                H[i, j] = 1 - H[i, j]
                new_error = np.sum((X - np.dot(W, H))**2)
                old_error = np.sum(X_diff**2)
                
                if new_error >= old_error:
                    H[i, j] = 1 - H[i, j]
                else:
                    X_diff = X - np.dot(W, H)
                    
        return W, H
    
    def fit_transform(self, X):
        """
        Fit the BMF model to X and return the transformed data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        Returns:
        --------
        W : array-like, shape (n_samples, n_components)
            Transformed data.
        """
        # Convert input to binary
        X = (X > 0).astype(int)
        
        # Initialize factors
        self._initialize_factors(X)
        
        # Iterate until convergence
        prev_error = float('inf')
        for _ in trange(self.max_iter, desc="BMF", leave=False):
            self.W, self.H = self._binary_mult_update(X, self.W, self.H)
            
            # Check convergence
            error = np.sum((X - np.dot(self.W, self.H))**2)
            if abs(prev_error - error) < self.tol:
                break
            prev_error = error
            
        return self.W
    
    def transform(self, X):
        """Transform new data using the fitted model."""
        return np.dot(X, self.H.T) > 0.5
    
    def reconstruct(self):
        """Reconstruct the matrix from factors."""
        return np.dot(self.W, self.H)