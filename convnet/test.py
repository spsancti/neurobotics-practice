from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
import layers
from layers import relu_forward
from layers import maxpool_forward


def main():
    image = misc.face()

    # normalize input image (kinda of)
    # do normalize every time!
    image = image / np.max(image)

    # sobel kernel here
    sobel = np.array(
    [
        [
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ],
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ],
            [
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ],
        ],
        [
            [
                [1,   2,  1],
                [0,   0,  0],
                [-1, -2, -1]
            ],
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ],
            [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ]
        ]
    ])

    # prepare image for convolution format
    # from HxWx3 to 1x3xHxW
    image1 = np.array([image])
    image1 = np.swapaxes(image1, 1, 3)
    image1 = np.swapaxes(image1, 2, 3)

    stuff = np.array(
        [
            [
                [
                    [.7]
                ],
                [
                    [.3]
                ]
            ],
            [
                [
                    [.3]
                ],
                [
                    [.7]
                ]
            ]
        ])


    # averaging kernel (1x1 convolution)
    avg = np.array(
        [
            [
                [
                    [.5]
                ],
                [
                    [.5]
                ]
            ]
        ])

    # do our "neural network"
    # basically, just convolutions
    image = image1

    image, cache = layers.conv_forward(image, sobel, 0)
    image, cache = layers.relu_forward(image)
    print(image.shape)
    # image, cache = maxpool_forward(image, size=2, stride=2)
    print(image.shape)

    image, cache = layers.conv_forward(image, stuff, 0, stride=1, padding=0)
    image, cache = layers.relu_forward(image)

    image, cache = maxpool_forward(image, size=2, stride=2)
    image = layers.maxpool_backward(image, cache)

    image, cache = layers.conv_forward(image, avg, 0)
    image, cache = layers.sigmoid_forward(image)


    # get rid of extra 1d dimensions for presentation purposes
    print(image.shape)
    image = image.squeeze()
    print(image.shape)

    # show this one!
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
