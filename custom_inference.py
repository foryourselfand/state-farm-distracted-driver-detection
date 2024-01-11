from time import perf_counter

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True, nogil=True, looplift=True, boundscheck=False, inline='always')
def conv_2d_relu(input_data, weights, biases):
    batch_size, input_height, input_width, input_channels = input_data.shape
    kernel_height, kernel_width, _, num_filters = weights.shape

    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    output_channels = num_filters

    output_data = np.zeros((batch_size, output_height, output_width, output_channels), dtype=np.float32)

    for b in prange(batch_size):
        for i in prange(output_height):
            for j in prange(output_width):
                for k in prange(num_filters):
                    for x in prange(kernel_height):
                        for y in prange(kernel_width):
                            for c in prange(input_channels):
                                output_data[b, i, j, k] += input_data[b, i + x, j + y, c] * weights[x, y, c, k]
                    output_data[b, i, j, k] += biases[k]
                    output_data[b, i, j, k] = max(0, output_data[b, i, j, k])
    return output_data


@njit(cache=True, fastmath=True, parallel=True, nogil=True, looplift=True, boundscheck=False, inline='always')
def max_pooling_2d(input_data, pool_height, pool_width):
    batch_size, input_height, input_width, num_channels = input_data.shape

    output_height = input_height // pool_height
    output_width = input_width // pool_width

    output_data = np.zeros((batch_size, output_height, output_width, num_channels), dtype=np.float32)

    for b in prange(batch_size):
        for i in prange(output_height):
            for j in prange(output_width):
                for c in prange(num_channels):
                    start_i = i * pool_height
                    start_j = j * pool_width
                    end_i = start_i + pool_height
                    end_j = start_j + pool_width

                    output_data[b, i, j, c] = np.max(input_data[b, start_i:end_i, start_j:end_j, c])

    return output_data


@njit(cache=True, fastmath=True, parallel=True, nogil=True, looplift=True, boundscheck=False, inline='always')
def dense_softmax_numba(input_data, weights, biases):
    batch_size, input_dim = input_data.shape
    output_dim = weights.shape[1]
    output_data = np.zeros((batch_size, output_dim), dtype=np.float32)
    for b in prange(batch_size):
        for i in prange(output_dim):
            for j in prange(input_dim):
                output_data[b, i] += input_data[b, j] * weights[j, i]
            output_data[b, i] += biases[i]
        output_exp = np.exp(output_data[b])
        output_sum = np.sum(output_exp)
        output_data[b] = output_exp / output_sum
    return output_data


def dense_softmax_numpy(input_data, weights, biases):
    output_data = np.dot(input_data, weights) + biases

    output_exp = np.exp(output_data)
    output_sum = np.sum(output_exp, axis=1, keepdims=True)
    output_data = output_exp / output_sum

    return output_data


def bootstrap_conv_2d_relu():
    input_data = np.random.rand(2, 3, 3, 1).astype("float32")
    conv2d_1_weights = np.random.rand(3, 3, 1, 1).astype("float32")
    conv2d_1_biases = np.random.rand(1).astype("float32")

    conv2d_1_output = conv_2d_relu(input_data, conv2d_1_weights, conv2d_1_biases)
    return len(conv2d_1_output)


def bootstrap_max_pooling_2d():
    input_data = np.random.rand(2, 2, 2, 1).astype("float32")
    max_pooling_2d_1_output = max_pooling_2d(input_data, 2, 2)
    return len(max_pooling_2d_1_output)


def bootstrap_dense_softmax():
    input_data = np.random.rand(2, 2).astype("float32")
    dense_weights = np.random.rand(2, 2).astype("float32")
    dense_biases = np.random.rand(2).astype("float32")

    dense_output = dense_softmax_numba(input_data, dense_weights, dense_biases)

    return len(dense_output)


def forward(input_data, conv_weights, conv_biases, dense_weights, dense_biases):
    conv_output = conv_2d_relu(input_data, conv_weights, conv_biases)
    max_pooling_output = max_pooling_2d(conv_output, 2, 2)
    flattened_output = max_pooling_output.reshape((max_pooling_output.shape[0], -1))
    dense_output = dense_softmax_numpy(flattened_output, dense_weights, dense_biases)
    return dense_output


def inference_test():
    start = perf_counter()
    bootstrap_conv_2d_relu()
    bootstrap_max_pooling_2d()
    bootstrap_dense_softmax()
    print(f'bootstrap: {perf_counter() - start:.2f}')

    start = perf_counter()
    data = np.load('data.npz')
    input_data = data['X_test']
    y_test = data['y_test']
    y_test = np.argmax(y_test, axis=1)
    print(f'load data: {perf_counter() - start:.2f}')
    print(f'input_data: {input_data.shape}')

    start = perf_counter()
    model_wb = np.load('SFDDD_model.npz')
    conv_weights = model_wb['conv_weights'].astype("float32")
    conv_biases = model_wb['conv_biases'].astype("float32")
    dense_weights = model_wb['dense_weights'].astype("float32")
    dense_biases = model_wb['dense_biases'].astype("float32")
    print(f'load model: {perf_counter() - start:.2f}')

    start = perf_counter()
    output = forward(input_data, conv_weights, conv_biases, dense_weights, dense_biases)
    output = np.argmax(output, axis=1)
    print(f'inference: {perf_counter() - start:.2f}')

    print(f'accuracy: {np.sum(output == y_test) / len(y_test):.2f}')


if __name__ == "__main__":
    inference_test()
