from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

class LogisticMatrixFactorization:
    def __init__(self, k=30, l2_p=1e-6, epochs=1000, learning_rate=0.001, tolerance=1e-4, device="cpu", random_state=None):
        """
        Logistic Matrix Factorization with a mask.

        Parameters:
        - k: Number of latent factors.
        - l2_p: Regularization parameter (L2 penalty).
        - epochs: Number of training epochs.
        - learning_rate: Learning rate for gradient descent.
        - tolerance: Early stopping criterion based on loss change.
        """
        self.k = k
        self.l2_p = l2_p
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.np = np
        self.random_state = random_state

        if device == "cpu":
            self.device = device
        elif device == "gpu":
            self.device = 0
        elif isinstance(device, int) and device >= 0:
            self.device = device
        else:
            raise Exception("Device should be 'cpu', 'gpu' (CUDA:0), or a GPU number between 0 and N-1 where N is the number of GPUs.")
        
        if self.device != "cpu" and cp is None:
            print("No CUDA found! Using CPU!")
            self.device = "cpu"

    def fit(self, Xtrain, MASK, plot_loss=True):
        """
        Train the logistic matrix factorization model.

        Parameters:
        - Xtrain: Training interaction matrix (m x n).
        - MASK: Binary mask matrix with 1s for observed entries in Xtrain.

        Returns:
        - W: Learned row (user) latent feature matrix (m x k).
        - H: Learned column (item) latent feature matrix (k x n).
        - row_bias: Learned row bias vector (m x 1).
        - col_bias: Learned column bias vector (1 x n).
        """
        if self.device != "cpu":
            self.np = cp

        m, n = Xtrain.shape
        W, H, row_bias, col_bias = self._initialize_embeddings(m, n)

        if self.device != "cpu":
            with cp.cuda.Device(self.device): 
                losses = cp.zeros(self.epochs)
                MASK = cp.array(MASK)
                Xtrain = cp.array(Xtrain)
                W, H, row_bias, col_bias, losses = self._factorization_routine(W, H, row_bias, col_bias, MASK, Xtrain, losses)

                # to CPU
                W = cp.asnumpy(W)
                H = cp.asnumpy(H)
                row_bias = cp.asnumpy(row_bias)
                col_bias = cp.asnumpy(col_bias)
                MASK = cp.asnumpy(MASK)
                Xtrain = cp.asnumpy(Xtrain)
                losses = cp.asnumpy(losses)
                self.np = np
        else:
            losses = np.zeros(self.epochs)
            W, H, row_bias, col_bias, losses = self._factorization_routine(W, H, row_bias, col_bias, MASK, Xtrain, losses)

        # Plot loss
        if plot_loss:
            plt.plot(losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.show()

        return W, H, row_bias, col_bias, losses

    def predict(self, W, H, row_bias, col_bias):
        """
        Predict all entries in the matrix.

        Parameters:
        - W: Learned row latent feature matrix (m x k).
        - H: Learned column latent feature matrix (k x n).
        - row_bias: Learned row bias vector (m x 1).
        - col_bias: Learned column bias vector (1 x n).

        Returns:
        - Xtilda: Predicted matrix of interaction probabilities.
        """
        return self._sigmoid(self.np.dot(W, H) + row_bias + col_bias)

    def map_probabilities_to_binary(self, Xtilda, threshold=0.5):
        """
        Map probabilities to binary values (0 or 1) using a threshold.

        Parameters:
        - Xtilda: numpy array, predicted probabilities (values in [0, 1]).
        - threshold: float, the cutoff for mapping probabilities to 0 or 1.

        Returns:
        - Xtilda_binary: numpy array, binary Xtilda (0s and 1s).
        """
        return (Xtilda >= threshold).astype(int)
    
    def _initialize_embeddings(self, m, n):
        """
        Initialize embeddings (W and H) and biases for rows (users) and columns (items).
        """
        np.random.seed(self.random_state)

        W = np.random.normal(scale=0.1, size=(m, self.k))
        H = np.random.normal(scale=0.1, size=(self.k, n))
        row_bias = np.random.normal(scale=0.1, size=(m, 1))
        col_bias = np.random.normal(scale=0.1, size=(1, n))

        if self.device != "cpu":
            with cp.cuda.Device(self.device): 
                W, H, row_bias, col_bias = cp.array(W), cp.array(H), cp.array(row_bias), cp.array(col_bias)

        return W, H, row_bias, col_bias

    def _sigmoid(self, x):
        return 1 / (1 + self.np.exp(-x))

    def _compute_loss(self, X_train, Xtilda, MASK, W, H):
        """
        Compute binary cross-entropy loss.

        Parameters:
        - X_train: Training interaction matrix.
        - Xtilda: Predicted matrix.
        - MASK: Binary mask matrix.

        Returns:
        - loss: Binary cross-entropy loss.
        """
        loss = -self.np.sum(
            MASK * (X_train * self.np.log(Xtilda + 1e-8) + (1 - X_train) * self.np.log(1 - Xtilda + 1e-8))
        )
        loss += self.l2_p * (self.np.sum(W ** 2) + self.np.sum(H ** 2))
        return loss


    def _factorization_routine(self, W, H, row_bias, col_bias, MASK, Xtrain, losses):
        """
        Performs matrix factorization using stochastic gradient descent (SGD) with regularization and optional early stopping.

        This function iteratively optimizes the latent factor matrices (`W` and `H`), row biases, and column biases 
        to minimize the reconstruction error between the observed entries in the input matrix (`Xtrain`) and the predicted 
        matrix (`Xtilda`). It incorporates L2 regularization and supports early stopping if the loss improvement falls 
        below a specified tolerance.

        Parameters:
            W (numpy.ndarray): 
                A matrix of shape `(num_rows, latent_factors)` representing the initial latent factors for rows.
            H (numpy.ndarray): 
                A matrix of shape `(latent_factors, num_columns)` representing the initial latent factors for columns.
            row_bias (numpy.ndarray): 
                A vector of shape `(num_rows, 1)` representing the row-wise biases.
            col_bias (numpy.ndarray): 
                A vector of shape `(1, num_columns)` representing the column-wise biases.
            MASK (numpy.ndarray): 
                A binary mask matrix of the same shape as `Xtrain`, where 1 indicates an observed entry and 0 indicates missing.
            Xtrain (numpy.ndarray): 
                The observed training data matrix of shape `(num_rows, num_columns)`.
            losses (list or numpy.ndarray): 
                A pre-allocated container to store the loss values at each epoch.

        Returns:
            W (numpy.ndarray): 
                The updated latent factor matrix for rows after optimization.
            H (numpy.ndarray): 
                The updated latent factor matrix for columns after optimization.
            row_bias (numpy.ndarray): 
                The updated row-wise biases.
            col_bias (numpy.ndarray): 
                The updated column-wise biases.
            losses (list or numpy.ndarray): 
                The updated list or array containing the training loss at each epoch.

        Steps:
        1. **Prediction**: The predicted matrix (`Xtilda`) is computed using the current `W`, `H`, `row_bias`, and `col_bias`.
        2. **Error Calculation**: The reconstruction error is calculated only for observed entries using the binary mask (`MASK`).
        3. **Gradient Calculation**: Gradients for `W`, `H`, `row_bias`, and `col_bias` are computed using the observed errors 
        and L2 regularization.
        4. **Parameter Updates**: The latent factor matrices (`W`, `H`) and biases (`row_bias`, `col_bias`) are updated using 
        the gradients and a specified learning rate.
        5. **Loss Calculation**: The reconstruction loss is computed for the current epoch and stored in the `losses` array.
        6. **Early Stopping**: If the loss improvement between consecutive epochs falls below a predefined tolerance, the 
        optimization process terminates early.

        Notes:
        - The `_compute_loss` function is assumed to compute the loss using both observed reconstruction errors and regularization terms.
        - Early stopping can significantly reduce computation time when the optimization converges quickly.
        - The function updates the input parameters in-place, and the returned values reflect the final state after optimization.
        """
        for epoch in tqdm(range(self.epochs)):
            # Compute Xtilda (predictions)
            Xtilda = self.predict(W, H, row_bias=row_bias, col_bias=col_bias)

            # Compute errors for observed entries
            errors = MASK * (Xtilda - Xtrain)

            # Gradients
            grad_W = self.np.dot(errors, H.T) + self.l2_p * W
            grad_H = self.np.dot(W.T, errors) + self.l2_p * H
            grad_row_bias = self.np.sum(errors, axis=1, keepdims=True) + self.l2_p * row_bias
            grad_col_bias = self.np.sum(errors, axis=0, keepdims=True) + self.l2_p * col_bias

            # Update embeddings and biases
            W -= self.learning_rate * grad_W
            H -= self.learning_rate * grad_H
            row_bias -= self.learning_rate * grad_row_bias
            col_bias -= self.learning_rate * grad_col_bias

            # Compute training loss
            loss = self._compute_loss(Xtrain, Xtilda, MASK, W, H)
            losses[epoch] = loss

            # Early stopping based on tolerance
            if self.tolerance is not None and (epoch > 0 and abs(losses[epoch] - losses[epoch-1]) < self.tolerance):
                print(f"Early stopping at epoch {epoch + 1}. Loss change below tolerance.")
                break
        
        return W, H, row_bias, col_bias, losses

