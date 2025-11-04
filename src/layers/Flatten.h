#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
class FlattenLayer : public Layer {
   public:
    FlattenLayer(const LayerParams inParams, const LayerParams outParams)
        : Layer(inParams, outParams, LayerType::DENSE) {}  // Use DENSE type as closest match

    // Virtual functions
    virtual void computeNaive(const LayerData& dataIn) const override;
    virtual void computeThreaded(const LayerData& dataIn) const override;
    virtual void computeTiled(const LayerData& dataIn) const override;
    virtual void computeSIMD(const LayerData& dataIn) const override;
};

}  // namespace ML