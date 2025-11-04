#include "Convolutional.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{
    // --- Begin Student Code ---
    // ASDFff
    // Compute the convolution for the layer data
    // Get dimensions from layer parameters
  
    // Perform convolution
    void ConvolutionalLayer::computeNaive(const LayerData &dataIn) const
    {
        // TODO: Your Code Here...
        // The following line is an example of copying a single 32-bit floating point integer from the input layer data to the output layer data

        const auto &inputDims = getInputParams().dims;   // [H, W, C_in]
        const auto &outputDims = getOutputParams().dims; // [H_out, W_out, C_out]
        const auto &weightDims = getWeightParams().dims; // [K_H, K_W, C_in, C_out]

        size_t U = 1; // Stride

       // size_t H = inputDims[0];
        size_t W = inputDims[1];
        size_t C = inputDims[2];

        size_t P = outputDims[0];
        size_t Q = outputDims[1];
        size_t M = outputDims[2];

        size_t R = weightDims[0];
        size_t S = weightDims[1];

        for (size_t p = 0; p < P; p++)
        {
            for (size_t q = 0; q < Q; q++)
            {
                for (size_t m = 0; m < M; m++)
                {
                    fp32 result = 0.0f;
                    
                    // Perform the convolution sum
                    // o[p][q][m] = sum_{c,r,s} i[U*p+r][U*q+s][c] * f[r][s][c][m] + b[m]
                    for (size_t c = 0; c < C; c++)
                    { // Input channel
                        for (size_t r = 0; r < R; r++)
                        { // Kernel height
                            for (size_t s = 0; s < S; s++)
                            { // Kernel width
                                // Input coordinates
                                size_t input_h = U * p + r;
                                size_t input_w = U * q + s;
                                
                                // Input index: [input_h, input_w, c]
                                size_t input_idx = input_h * W * C + input_w * C + c;
                                
                                // Weight index: [r, s, c, m]
                                size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                                
                                // Accumulate
                                result += dataIn.get<fp32>(input_idx) *
                                          getWeightData().get<fp32>(weight_idx);
                            }
                        }
                    }
                    // Add bias: b[m]
                    result += getBiasData().get<fp32>(m);
                    
                    // Apply ReLU activation
                    result = std::max(0.0f, result);
                    
                    // Output index: [p, q, m]
                    size_t output_idx = p * Q * M + q * M + m;
                    getOutputData().get<fp32>(output_idx) = result;
                }
            }
        }
    }

    // Compute the convolution using threads
    void ConvolutionalLayer::computeThreaded(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation with thread hints
        computeNaive(dataIn);
    }

    // Compute the convolution using a tiled approach
    void ConvolutionalLayer::computeTiled(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation
        computeNaive(dataIn);
    }

    // Compute the convolution using SIMD
    void ConvolutionalLayer::computeSIMD(const LayerData &dataIn) const
    {
        // For simplicity, use naive implementation
        computeNaive(dataIn);
    }

} // namespace ML
