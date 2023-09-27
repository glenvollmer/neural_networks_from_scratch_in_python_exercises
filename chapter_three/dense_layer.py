from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt
import nnfs

nnfs.init()

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

def main():
    x, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 3)
    dense1.forward(x)

    print(dense1.output[:5])
    return

if __name__ == '__main__':
    main()