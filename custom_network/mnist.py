import sys
import os
import numpy as np
import cv2

def load_mnist_image_file(image_file):
    
    with open(image_file, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)

    return np.reshape(images, [-1, 28, 28])


def load_mnist_label_file(label_file):

    with open(label_file, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    return labels


def main():
    images = load_mnist_image_file('/home/raj/study/job_search/coding/deeplearning/data/mnist/t10k-images-idx3-ubyte')
    labels = load_mnist_label_file('/home/raj/study/job_search/coding/deeplearning/data/mnist/t10k-labels-idx1-ubyte')

    idx = 0
    for image, label in zip(images[:10], labels[:10]):

        idx += 1
        print('idx: %d label: %d'%(idx, label))
        cv2.imshow('image', image)    
        cv2.waitKey(0)


if __name__ == '__main__':
    exit(main())
    