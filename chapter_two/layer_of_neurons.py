
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



def main():
    n1 = first_neuron()
    
    print(f'first neuron: {n1}')
    return

if __name__ == '__main__':
    main()