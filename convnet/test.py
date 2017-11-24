from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
import layers
from layers import relu_forward


def main():
    image = misc.face()

    # normalize input image (kinda of)
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
    image = layers.conv_forward(image1, sobel, 0)
    image = relu_forward(image)
    image = layers.conv_forward(image, avg, 0)
    image = relu_forward(image)

    # get rid of extra 1d dimensions for presentation purposes
    print(image.shape)
    image = image.squeeze()
    print(image.shape)

    # show this one!
    plt.imshow(image, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
