import numpy as np

def first_neuron():
    '''
        A single neuron with three inputs, three weights, and a bias
    '''
    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    output = np.dot(weights, inputs) + bias

    return output


def main():
    n1 = first_neuron()
    
    print(f'first neuron: {n1}')
    return

if __name__ == '__main__':
    main()