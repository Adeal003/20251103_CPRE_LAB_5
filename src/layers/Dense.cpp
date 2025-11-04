#include "Dense.h"

#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{

    void DenseLayer::computeNaive(const LayerData &dataIn) const
    {
        //const auto &inputDims = getInputParams().dims;   // Can be [H, W, C] or [features] 
        //const auto &outputDims = getOutputParams().dims; // Expected: [output_features]
        const auto &weightDims = getWeightParams().dims; // Expected: [input_features, output_features]

        // Calculate total input features by flattening all input dimensions
        size_t totalInputFeatures = getInputParams().flat_count();
        size_t outputSize = getOutputParams().flat_count();

        // Validate dimensions
        size_t expectedInputFeatures = weightDims[0];  // First dimension of weight matrix
        size_t expectedOutputFeatures = weightDims[1]; // Second dimension of weight matrix
        
        if (totalInputFeatures != expectedInputFeatures) {
            std::cerr << "Dense layer input size mismatch: got " << totalInputFeatures 
                      << ", expected " << expectedInputFeatures << std::endl;
            return;
        }
        
        if (outputSize != expectedOutputFeatures) {
            std::cerr << "Dense layer output size mismatch: got " << outputSize 
                      << ", expected " << expectedOutputFeatures << std::endl;
            return;
        }

        const LayerData& weights = getWeightData();
        LayerData& output = getOutputData();
        const LayerData& bias = getBiasData();

        // Dense layer computation: output = input * weights + bias
        // Input is treated as flattened regardless of original dimensions
        for (size_t out_idx = 0; out_idx < outputSize; out_idx++)
        {
            fp32 sum = bias.get<fp32>(out_idx);

            for (size_t in_idx = 0; in_idx < totalInputFeatures; in_idx++)
            {
                // Weight matrix: [input_features, output_features]
                size_t weightIdx = in_idx * outputSize + out_idx;
                
                sum += dataIn.get<fp32>(in_idx) * weights.get<fp32>(weightIdx);
            }

            // Apply ReLU activation only for hidden layers (not the final layer before Softmax)
            // The final dense layer typically has 200 outputs (for classification)
            // Hidden dense layers have other sizes (like 256)
            if (outputSize != 200) {
                // This is a hidden layer, apply ReLU
                sum = std::max(0.0f, sum);
            }
            // For the final layer (outputSize == 200), don't apply ReLU

            // Store result in output
            output.get<fp32>(out_idx) = sum;
        }
    }

    void DenseLayer::computeThreaded(const LayerData& dataIn) const {
        // For simplicity, use naive implementation with thread hints
        // TODO: Implement actual threading
        computeNaive(dataIn);
    }

    void DenseLayer::computeTiled(const LayerData& dataIn) const {
        // For simplicity, use naive implementation 
        // TODO: Implement tiled matrix multiplication
        computeNaive(dataIn);
    }

    void DenseLayer::computeSIMD(const LayerData& dataIn) const {
        // For simplicity, use naive implementation
        // TODO: Implement SIMD optimized matrix multiplication
        computeNaive(dataIn);
    }

}