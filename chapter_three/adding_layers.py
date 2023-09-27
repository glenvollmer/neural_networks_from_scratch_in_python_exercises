import numpy as np

def numpy_layers():
    inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ]
    
    weights = [
        [0.2, 0.8, -0.5, 1.0], 
        [0.5, -0.91, 0.26, -0.5], 
        [-0.26, -0.27, 0.17, 0.87]
    ]
    
    biases = [2.0, 3.0, 0.5]

    weights2 = [
        [0.1, -0.14, 0.5],
        [-0.5, 0.12, -0.33],
        [-0.44, 0.73, -0.13]
    ]
    
    biases2 = [-1.0, 2.0, -0.5]

    transposed_weights = np.array(weights).T
    layer1_outputs = np.dot(inputs, transposed_weights) + biases

    transposed_weights2 = np.array(weights2).T

    layer2_outputs = np.dot(layer1_outputs, transposed_weights2) + biases2

    return layer2_outputs




def main():
    nl = numpy_layers()
    
    print(f'numpy layers output: {nl}')
    return

if __name__ == '__main__':
    main()