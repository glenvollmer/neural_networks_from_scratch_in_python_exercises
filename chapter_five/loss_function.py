from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt
import nnfs

nnfs.init()

# calculate loss on output probabilities
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class LossCategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_true_shape = len(y_true.shape)

        if y_true_shape == 1:
            sample_range = range(samples)
            correct_confidences = y_pred_clipped[sample_range, y_true]
        
        elif y_true_shape == 2:
            y_pred_clip_true_product = y_pred_clipped * y_true
            correct_confidences = np.sum(y_pred_clip_true_product, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods
    

# layers for our networks  
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# activation functions for each layer output
class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# loss functions


def main():
    x, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()
    loss_function = LossCategoricalCrossentropy()

    dense1.forward(x)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_function.calculate(activation2.output, y)


    print(activation2.output[:5])
    print('\n')
    print(f'loss: {loss}')
    return

if __name__ == '__main__':
    main()