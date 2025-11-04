#pragma once
#include <vector>
#include <memory>

#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"

namespace ML {
class Model {
   public:
    // Constructors
    inline Model() : layers() {}  //, checkFinal(true), checkEachLayer(false) {}

    // Functions
    const LayerData& inference(const LayerData& inData, const Layer::InfType infType = Layer::InfType::NAIVE) const;
    const LayerData& inferenceLayer(const LayerData& inData, const int layerNum, const Layer::InfType infType = Layer::InfType::NAIVE) const;

    // Internal memory management
    // Allocate the internal output buffers for each layer in the model
    inline void allocLayers();

    // Free all layers
    inline void freeLayers();

    // Getter Functions
    inline const std::size_t getNumLayers() const { return layers.size(); }

    // Add a layer to the model
    template<typename T, typename... Args> void addLayer(Args&&... args) { layers.emplace_back(new T(std::forward<Args>(args)...)); }

    // Insert a layer into the model
    // void insertLayer(Layer* l, std::size_t idx) { layers.insert(layers.begin() + idx, l); }

    // Remove a layer from the model
    inline void removeLayer(const std::size_t idx) { layers.erase(layers.begin() + idx); }

    // Get layer from the model
    inline Layer& getLayer(const std::size_t idx) { return *layers[idx]; }
    inline const Layer& getLayer(const std::size_t idx) const { return *layers[idx]; }

    // Get the last layer from the model
    inline Layer& getOutputLayer() { return *layers[layers.size() - 1]; }
    inline const Layer& getOutputLayer() const { return *layers[layers.size() - 1]; }

    // Array operator (get the layer index)
    inline Layer& operator[](const std::size_t idx) { return *layers[idx]; }
    inline const Layer& operator[](const std::size_t idx) const { return *layers[idx]; }

    // Call operators (run inference)
    inline const LayerData& operator()(const LayerData& inData, const Layer::InfType infType = Layer::InfType::NAIVE) const {
        return inference(inData, infType);
    }
    inline const LayerData& operator()(const LayerData& inData, const int layerNum, const Layer::InfType infType = Layer::InfType::NAIVE) const {
        return inferenceLayer(inData, layerNum, infType);
    }

   private:
    std::vector<std::unique_ptr<Layer>> layers;
};

// Allocate the internal output buffers for each layer in the model
void Model::allocLayers() {
    for (std::size_t i = 0; i < layers.size(); i++) {
        layers[i]->allocLayer();
    }
}

// Free all layers in the model
void Model::freeLayers() {
    // All classes use RAII, so just wipe out the vector of layers.
    layers.clear();
}
}  // namespace ML
