#include "Flatten.h"

#include <iostream>
#include <cstring>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML
{

    void FlattenLayer::computeNaive(const LayerData &dataIn) const
    {
        //const auto &inputDims = getInputParams().dims;
        //const auto &outputDims = getOutputParams().dims;

        // Validate that input and output have the same total number of elements
        size_t inputElements = getInputParams().flat_count();
        size_t outputElements = getOutputParams().flat_count();
        
        if (inputElements != outputElements) {
            std::cerr << "Flatten layer element count mismatch: input=" 
                      << inputElements << ", output=" << outputElements << std::endl;
            return;
        }

        LayerData& output = getOutputData();
        
        // Simply copy the data (flattening is just a reshape operation)
        std::memcpy(output.raw(), dataIn.raw(), inputElements * sizeof(fp32));
    }

    void FlattenLayer::computeThreaded(const LayerData& dataIn) const {
        // Flattening is just memory copy, no threading needed
        computeNaive(dataIn);
    }

    void FlattenLayer::computeTiled(const LayerData& dataIn) const {
        // Flattening is just memory copy, no tiling needed
        computeNaive(dataIn);
    }

    void FlattenLayer::computeSIMD(const LayerData& dataIn) const {
        // Flattening is just memory copy, no SIMD needed
        computeNaive(dataIn);
    }

}