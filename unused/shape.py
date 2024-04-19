from sklearn import datasets
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    X, y = datasets.make_moons(1000, noise=0.02)
    plt.plot(X[:, 0], X[:, 1], 'ro')
    plt.show()

    X, y = datasets.make_circles(1000, noise=0.02)
    plt.plot(X[:, 0], X[:, 1], 'ro')
    plt.show()

    X, _ = datasets.make_swiss_roll(1000, noise=0.2)
    X = X[:, [0, 2]]/10.0
    plt.plot(X[:, 0], X[:, 1], 'ro')
    plt.show()

    X, _ = datasets.make_s_curve(1000, noise=0.2)
    X = X[:, [0, 2]]/10.0
    plt.plot(X[:, 0], X[:, 1], 'ro')
    plt.show()

    dataset = torch.Tensor(X).float()
    print(dataset.shape)
