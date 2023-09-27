from nnfs.datasets import spiral_data

import matplotlib.pyplot as plt
import nnfs

nnfs.init()


def main():
    x, y = spiral_data(samples=100, classes=3)
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
    plt.show()    
    return

if __name__ == '__main__':
    main()