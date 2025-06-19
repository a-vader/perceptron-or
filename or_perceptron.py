import numpy as np

class Perceptron:
    def __init__(self):
        self.weights = np.zeros(3)

    def predict(self, x):
        x = np.insert(x, 0, 1)
        return int(np.dot(self.weights, x) >= 0)

    def train(self, X, y, lr=1.0, epochs=10):
        for _ in range(epochs):
            for xi, target in zip(X, y):
                x_aug = np.insert(xi, 0, 1)
                output = self.predict(xi)
                self.weights += lr * (target - output) * x_aug

# Данные для OR
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 1, 1, 1])

model = Perceptron()
model.train(X, y)

print("=== OR ===")
for x in X:
    print(f"{x} -> {model.predict(x)}")
