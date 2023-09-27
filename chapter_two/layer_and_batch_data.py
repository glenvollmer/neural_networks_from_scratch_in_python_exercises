import numpy as np

def numpy_layer():

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
    transposed_weights = np.array(weights).T

    layer_outputs = np.dot(inputs, transposed_weights) + biases

    return layer_outputs




def main():
    nl = numpy_layer()
    
    print(f'numpy layer: {nl}')
    return

if __name__ == '__main__':
    main()