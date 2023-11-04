import numpy as np

class Sgd:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate


    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.momentum_rate * self.v) - (self.learning_rate * gradient_tensor)
        weight_tensor = weight_tensor + self.v
        return weight_tensor

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k_iterCounter = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.mu * self.v) + (1 - self.mu) * gradient_tensor
        self.r = (self.rho * self.r) + (1 - self.rho) * np.square(gradient_tensor)
        self.k_iterCounter = self.k_iterCounter+1
        v_hat = self.v / (1 - self.mu ** self.k_iterCounter)
        r_hat = self.r / (1 - self.rho ** self.k_iterCounter)
        weight_tensor = weight_tensor - self.learning_rate * v_hat / (np.sqrt(r_hat) + np.finfo(float).eps)
        return weight_tensor