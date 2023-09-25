import numpy as np

def matrix_product_example():
    '''
        create a matrix product of a row vector and a column vector
    '''

    a = [1, 2, 3]
    b = [2, 3, 4]
    
    a = np.array([a])
    b = np.array([b]).T

    matrix_product = np.dot(a, b)

    return matrix_product




def main():
    mp = matrix_product_example()
    
    print(f'matrix product example: {mp}')
    return

if __name__ == '__main__':
    main()