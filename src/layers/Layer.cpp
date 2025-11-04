#include "Layer.h"

#include <cassert>
#include <iostream>
#include <vector>

#include "../Utils.h"

namespace ML {
// Ensure that Layer params are compatible
bool LayerParams::isCompatible(const LayerParams& params) const {
    if (elementSize != params.elementSize) throw std::runtime_error("Element Size of params must match");
    if (dims.size() != params.dims.size()) throw std::runtime_error("Must have the same number of dimentions");
    for (std::size_t i = 0; i < dims.size(); i++) {
        if (dims[i] != params.dims[i]) throw std::runtime_error("Each dimention must match");
        if (dims[i] != params.dims[i]) return false;
    }

    return elementSize == params.elementSize && dims.size() == params.dims.size();
}

// Ensure that data being inputted is of the correct size and shape that the layer expects
bool Layer::checkDataInputCompatibility(const LayerData& data) const { return inParams.isCompatible(data.getParams()); }

}  // namespace ML
