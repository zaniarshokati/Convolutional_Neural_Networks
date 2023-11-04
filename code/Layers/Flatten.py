import numpy as np


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = np.shape(input_tensor)
        batch_size = self.input_shape[0]
        matrix_size = np.size(input_tensor)
        col_size = int(matrix_size / batch_size)
        output = np.reshape(input_tensor, (batch_size, col_size))
        return output

    def backward(self, error_tensor):
        output = np.reshape(error_tensor, self.input_shape)
        return output
