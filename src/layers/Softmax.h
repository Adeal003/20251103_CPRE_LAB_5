#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
class SoftmaxLayer : public Layer {
   public:
    SoftmaxLayer(const LayerParams inParams, const LayerParams outParams)
        : Layer(inParams, outParams, LayerType::SOFTMAX) {}

    // Allocate all resources needed for the layer
    virtual void allocLayer() override {
        Layer::allocLayer();
        // Softmax doesn't need to load additional data
    }

    // Free all resources allocated for the layer
    virtual void freeLayer() override {
        Layer::freeLayer();
        // Softmax doesn't have additional resources to free
    }

    // Virtual functions
    virtual void computeNaive(const LayerData& dataIn) const override;
    virtual void computeThreaded(const LayerData& dataIn) const override;
    virtual void computeTiled(const LayerData& dataIn) const override;
    virtual void computeSIMD(const LayerData& dataIn) const override;

   private:
    // Softmax doesn't have additional parameters
};

}  // namespace ML