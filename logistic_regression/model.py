import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def weight_initialization(self, X):
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0

    def sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def compute_gradients(self, X, y, y_pred):
        difference = y_pred - y
        dw = (1 / self.num_samples) * np.dot(X.T, difference)
        db = (1 / self.num_samples) * np.sum(difference)
        return dw, db

    def compute_loss(self, y_true, y_pred):
        # Binary cross entropy loss (Log loss)
        loss = - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
        return np.mean(loss)

    def fit(self, X, y):
        # Define variables
        self.num_samples, self.num_features = X.shape
        train_loss_list = []
        train_acc_list = []
        parts = int(self.epochs // 10)

        # Initialize weights
        self.weight_initialization(X)

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(y_pred)

            # Calculate accuracy
            y_pred_list = np.array([1 if y_pred[i] > 0.5 else 0 for i in range(len(y_pred))])
            acc = accuracy_score(y, y_pred_list)
            train_acc_list.append(acc)

            # Calculate loss
            loss = self.compute_loss(y, y_pred)
            train_loss_list.append(loss)

            # Update model parameters
            dw, db = self.compute_gradients(X, y, y_pred)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # if epoch % parts == 0:
            #     print(f"Epoch {epoch}: Train accuracy: {acc} \t Train loss: {loss}")
        
        train_acc = np.mean(train_acc_list)
        train_loss = np.mean(train_loss_list)

        return train_acc, train_loss

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(y_pred)
        return [1 if y_pred[i] > 0.5 else 0 for i in range(len(y_pred))]