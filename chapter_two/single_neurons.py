
def first_neuron():
    '''
        A single neuron with three inputs, three weights, and a bias
    '''
    inputs = [1, 2, 3]
    weights = [0.2, 0.8, -0.5]
    bias = 2

    output = (inputs[0] * weights[0] + 
              inputs[1] * weights[1] + 
              inputs[2] * weights[2] + bias)
    return output


def second_neuron():
    '''
        A single neuron with four inputs, four weights, and a bias
    '''

    inputs = [1.0, 2.0, 3.0, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    output = (inputs[0] * weights[0] + 
              inputs[1] * weights[1] + 
              inputs[2] * weights[2] + 
              inputs[3] * weights[3] +
              bias)
    return output

def main():
    n1 = first_neuron()
    n2 = second_neuron()
    
    print(f'first neuron: {n1}')
    print(f'second neuron: {n2}')
    return

if __name__ == '__main__':
    main()