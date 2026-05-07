#include <math.h>
#include "kernels.h"

__global__ void forward_relu_layer_kernel(const float *input, const float *weights, const float *bias, float *pre_activation, float *activation, int input_size, int output_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= output_size) return;

    float sum = bias[j];
    for (int i = 0; i < input_size; i++) {
        sum += input[i] * weights[i * output_size + j];
    }

    if (pre_activation) pre_activation[j] = sum;
    activation[j] = sum > 0.0f ? sum : 0.0f;
}

__global__ void forward_linear_layer_kernel(const float *input, const float *weights, const float *bias, float *output, int input_size, int output_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= output_size) return;

    float sum = bias[j];
    for (int i = 0; i < input_size; i++) {
        sum += input[i] * weights[i * output_size + j];
    }

    output[j] = sum;
}

__global__ void softmax_output_delta_kernel(const float *logits, const float *labels, float *probabilities, float *delta, float *loss, int classes) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    float max_value = logits[0];
    for (int k = 1; k < classes; k++) {
        if (logits[k] > max_value) max_value = logits[k];
    }

    float sum = 0.0f;
    for (int k = 0; k < classes; k++) {
        probabilities[k] = expf(logits[k] - max_value);
        sum += probabilities[k];
    }

    float sample_loss = 0.0f;
    for (int k = 0; k < classes; k++) {
        probabilities[k] /= sum;
        delta[k] = labels[k] - probabilities[k];
        sample_loss -= labels[k] * logf(probabilities[k] + 1e-8f);
    }

    if (loss) atomicAdd(loss, sample_loss);
}

__global__ void backprop_hidden_delta_kernel(const float *next_delta, const float *next_weights, const float *activation, float *delta, int current_size, int next_size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= current_size) return;

    float err = 0.0f;
    for (int k = 0; k < next_size; k++) {
        err += next_delta[k] * next_weights[j * next_size + k];
    }

    delta[j] = activation[j] > 0.0f ? err : 0.0f;
}

__global__ void update_weight_matrix_kernel(float *weights, const float *input_activation, const float *delta, float learning_rate, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = input_size * output_size;
    if (idx >= total) return;

    int i = idx / output_size;
    int j = idx % output_size;
    weights[idx] += learning_rate * input_activation[i] * delta[j];
}

__global__ void update_bias_vector_kernel(float *bias, const float *delta, float learning_rate, int size) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= size) return;

    bias[j] += learning_rate * delta[j];
}
