from network import Network
import numpy as np
import mnist

ip_size = 784
n_class = 10
# ip_size = 5
# n_class = 5
layer_size = [ip_size, 10, 20, 10, n_class]
lr = 0.001
image_file = '/home/raj/study/job_search/coding/deeplearning/data/mnist/t10k-images-idx3-ubyte'
label_file = '/home/raj/study/job_search/coding/deeplearning/data/mnist/t10k-labels-idx1-ubyte'
n_batch = 100000
batch_size = 16

nw = Network(layer_size=layer_size, activation_fn='relu', learning_rate=lr,
gradient_method='default', loss_layer_activation_fn='softmax', loss_func='cross_entropy')

examples = mnist.load_mnist_image_file(image_file)/255
labels = mnist.load_mnist_label_file(label_file)
# labels = np.random.choice(5, 1000)
# examples = np.zeros((1000, 5), dtype=np.float32)
# examples[np.arange(1000), labels] = 1
indices = np.arange(len(examples))

for batch in range(n_batch):

    sample_idx = np.random.choice(indices, batch_size)
    samples = np.transpose(np.reshape(examples[sample_idx], [-1, ip_size]))
    gt = np.zeros((n_class, batch_size), dtype=np.float32)
    gt[labels[sample_idx], np.arange(batch_size)] = 1

    pred = nw.forward(np.array(samples, dtype=np.float32))
    loss = nw.compute_loss(pred=pred, gt=gt)
    nw.backward()
    nw.apply_gradient()

    print(batch+1, loss)

n_batch = 1000
nerrors = 0
for batch in range(n_batch):

    sample_idx = np.random.choice(indices, batch_size)
    samples = np.transpose(np.reshape(examples[sample_idx], [-1, ip_size]))
    gt = np.zeros((n_class, batch_size), dtype=np.float32)
    gt[labels[sample_idx], np.arange(batch_size)] = 1

    samples = np.array(samples, dtype=np.float32)
    pred = nw.forward(samples)
    loss = nw.compute_loss(pred=pred, gt=gt)
    nerrors += np.count_nonzero(labels[sample_idx] - np.argmax(pred, axis=0))

    print('--------------------------------')
    print('batch: ', batch+1)
    print('loss: ', loss)
    # print('samples: ', samples)
    print('gt: ', labels[sample_idx])
    print('pred: ', np.argmax(pred, axis=0))
    print('--------------------------------')    

n_examples = n_batch * batch_size
print('# examples: ', n_examples)
print('# errors: ', nerrors)
print('%s error: %.3f'%('%', 100*nerrors/n_examples) )