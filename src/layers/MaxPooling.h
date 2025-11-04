#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
class MaxPoolingLayer : public Layer {
   public:
    MaxPoolingLayer(const LayerParams inParams, const LayerParams outParams, const LayerParams poolParams)
        : Layer(inParams, outParams, LayerType::MAX_POOLING),
          poolParam(poolParams) {}

    // Getters
    const LayerParams& getPoolParams() const { return poolParam; }

    // Allocate all resources needed for the layer
    virtual void allocLayer() override {
        Layer::allocLayer();
        // MaxPooling doesn't need to load additional data
    }

    // Free all resources allocated for the layer
    virtual void freeLayer() override {
        Layer::freeLayer();
        // MaxPooling doesn't have additional resources to free
    }

    // Virtual functions
    virtual void computeNaive(const LayerData& dataIn) const override;
    virtual void computeThreaded(const LayerData& dataIn) const override;
    virtual void computeTiled(const LayerData& dataIn) const override;
    virtual void computeSIMD(const LayerData& dataIn) const override;

   private:
    LayerParams poolParam; // Stores pool size parameters [pool_h, pool_w]
};

}  // namespace ML