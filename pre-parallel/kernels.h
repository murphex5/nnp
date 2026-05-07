#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

__global__ void forward_relu_layer_kernel(const float *input, const float *weights, const float *bias, float *pre_activation, float *activation, int input_size, int output_size);
__global__ void forward_linear_layer_kernel(const float *input, const float *weights, const float *bias, float *output, int input_size, int output_size);
__global__ void softmax_output_delta_kernel(const float *logits, const float *labels, float *probabilities, float *delta, float *loss, int classes);
__global__ void backprop_hidden_delta_kernel(const float *next_delta, const float *next_weights, const float *activation, float *delta, int current_size, int next_size);
__global__ void update_weight_matrix_kernel(float *weights, const float *input_activation, const float *delta, float learning_rate, int input_size, int output_size);
__global__ void update_bias_vector_kernel(float *bias, const float *delta, float learning_rate, int size);

#endif
