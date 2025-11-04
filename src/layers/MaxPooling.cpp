#include "MaxPooling.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{

    void MaxPoolingLayer::computeNaive(const LayerData &dataIn) const
    {
        const auto &inputDims = getInputParams().dims;   // Expected: [H_in, W_in, C_in]
        const auto &outputDims = getOutputParams().dims; // Expected: [H_out, W_out, C_out]
        const auto &poolDims = getPoolParams().dims;     // Expected: [pool_h, pool_w]

        size_t inputHeight = inputDims[0];
        size_t inputWidth = inputDims[1];
        size_t inputChannels = inputDims[2];

        size_t outputHeight = outputDims[0];
        size_t outputWidth = outputDims[1];
        size_t outputChannels = outputDims[2];

        size_t poolHeight = poolDims[0];
        size_t poolWidth = poolDims[1];

        LayerData& output = getOutputData();

        // Max pooling computation
        for (size_t c = 0; c < outputChannels; c++)
        {
            for (size_t h_out = 0; h_out < outputHeight; h_out++)
            {
                for (size_t w_out = 0; w_out < outputWidth; w_out++)
                {
                    fp32 maxVal = -INFINITY;

                    // Pool over the kernel region
                    for (size_t pool_h = 0; pool_h < poolHeight; pool_h++)
                    {
                        for (size_t pool_w = 0; pool_w < poolWidth; pool_w++)
                        {
                            size_t h_in = h_out * poolHeight + pool_h;
                            size_t w_in = w_out * poolWidth + pool_w;

                            // Check bounds
                            if (h_in < inputHeight && w_in < inputWidth)
                            {
                                size_t inputIdx = h_in * (inputWidth * inputChannels) +
                                                  w_in * inputChannels +
                                                  c;

                                fp32 val = dataIn.get<fp32>(inputIdx);
                                if (val > maxVal)
                                {
                                    maxVal = val;
                                }
                            }
                        }
                    }

                    size_t outputIdx = h_out * (outputWidth * outputChannels) +
                                       w_out * outputChannels +
                                       c;
                    
                    output.get<fp32>(outputIdx) = maxVal;
                }
            }
        }
    }

    void MaxPoolingLayer::computeThreaded(const LayerData& dataIn) const {
        // For simplicity, use naive implementation with thread hints
        // TODO: Implement actual threading
        computeNaive(dataIn);
    }

    void MaxPoolingLayer::computeTiled(const LayerData& dataIn) const {
        // For simplicity, use naive implementation 
        // TODO: Implement tiled processing
        computeNaive(dataIn);
    }

    void MaxPoolingLayer::computeSIMD(const LayerData& dataIn) const {
        // For simplicity, use naive implementation
        // TODO: Implement SIMD optimized max pooling
        computeNaive(dataIn);
    }

}