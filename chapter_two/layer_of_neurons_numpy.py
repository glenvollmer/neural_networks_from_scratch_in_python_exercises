import numpy as np

def numpy_layer():
    '''
        Same single layer of neurons with numpy dot multiplication
    '''

    inputs = [1, 2, 3, 2.5]
    
    weights = [
        [0.2, 0.8, -0.5, 1], 
        [0.5, -0.91, 0.26, -0.5], 
        [-0.26, -0.27, 0.17, 0.87]
        ]
    
    biases = [2, 3, 0.5]

    layer_outputs = np.dot(weights, inputs) + biases

    return layer_outputs




def main():
    nl = numpy_layer()
    
    print(f'numpy layer: {nl}')
    return

if __name__ == '__main__':
    main()