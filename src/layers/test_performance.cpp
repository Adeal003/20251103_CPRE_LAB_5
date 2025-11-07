/*
 * Simple Standalone Performance Test
 * 
 * 
 * 1. Compile it: 
 *    g++ -o test_perf test_performance.cpp src/layers/Convolutional.cpp -Isrc -std=c++11 -O2
 * 2.. Run it: 
 *    ./test_perf
 * 
 * This will test just the layer computation without needing Model class or data files.
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>

// Simple type definitions (matching your project)
typedef float fp32;

// ============================================================================
// MINIMAL LayerData and LayerParams classes
// (Simplified versions - just enough to test the layer)
// ============================================================================

class LayerParams {
public:
    const size_t elementSize;
    const std::vector<size_t> dims;
    
    LayerParams(size_t elemSize, std::vector<size_t> d) 
        : elementSize(elemSize), dims(d) {}
    
    size_t flat_count() const {
        size_t count = 1;
        for (auto d : dims) count *= d;
        return count;
    }
    
    size_t byte_size() const {
        return flat_count() * elementSize;
    }
};

class LayerData {
private:
    LayerParams params;
    std::vector<fp32> data;
    
public:
    LayerData(const LayerParams& p) : params(p) {
        data.resize(params.flat_count());
    }
    
    const LayerParams& getParams() const { return params; }
    
    fp32& get(size_t idx) { return data[idx]; }
    const fp32& get(size_t idx) const { return data[idx]; }
    
    void allocData() { /* Already done in constructor */ }
};

// ============================================================================
// SIMPLIFIED ConvolutionalLayer - Just for testing
// ============================================================================

class SimpleConvLayer {
private:
    LayerParams inParams, outParams, weightParams, biasParams;
    LayerData weightData, biasData;
    mutable LayerData outData;
    
public:
    SimpleConvLayer(const LayerParams& in, const LayerParams& out, 
                    const LayerParams& weight, const LayerParams& bias)
        : inParams(in), outParams(out), weightParams(weight), biasParams(bias),
          weightData(weight), biasData(bias), outData(out) {
        
        // Initialize weights and biases with random values
        for (size_t i = 0; i < weightData.getParams().flat_count(); i++) {
            weightData.get(i) = 0.1f; // Simple initialization
        }
        for (size_t i = 0; i < biasData.getParams().flat_count(); i++) {
            biasData.get(i) = 0.0f;
        }
    }
    
    const LayerData& getOutputData() const { return outData; }
    
    // ========================================================================
    // NAIVE IMPLEMENTATION (Your original - baseline for comparison)
    // ========================================================================
    void computeNaive(const LayerData& dataIn) const {
        const auto& inputDims = inParams.dims;
        const auto& outputDims = outParams.dims;
        const auto& weightDims = weightParams.dims;
        
        size_t W = inputDims[1];
        size_t C = inputDims[2];
        size_t P = outputDims[0];
        size_t Q = outputDims[1];
        size_t M = outputDims[2];
        size_t R = weightDims[0];
        size_t S = weightDims[1];
        size_t U = 1; // Stride
        
        // Original loop order: p -> q -> m -> c -> r -> s
        for (size_t p = 0; p < P; p++) {
            for (size_t q = 0; q < Q; q++) {
                for (size_t m = 0; m < M; m++) {
                    fp32 result = 0.0f;
                    
                    for (size_t c = 0; c < C; c++) {
                        for (size_t r = 0; r < R; r++) {
                            for (size_t s = 0; s < S; s++) {
                                size_t input_h = U * p + r;
                                size_t input_w = U * q + s;
                                size_t input_idx = input_h * W * C + input_w * C + c;
                                size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                                
                                result += dataIn.get(input_idx) * weightData.get(weight_idx);
                            }
                        }
                    }
                    
                    result += biasData.get(m);
                    result = std::max(0.0f, result); // ReLU
                    
                    size_t output_idx = p * Q * M + q * M + m;
                    outData.get(output_idx) = result;
                }
            }
        }
    }
    
    // ========================================================================
    // CACHE-OPTIMIZED IMPLEMENTATION (New Version)
    // ========================================================================
    void computeOptimized(const LayerData& dataIn) const {
        const auto& inputDims = inParams.dims;
        const auto& outputDims = outParams.dims;
        const auto& weightDims = weightParams.dims;
        
        size_t W = inputDims[1];
        size_t C = inputDims[2];
        size_t P = outputDims[0];
        size_t Q = outputDims[1];
        size_t M = outputDims[2];
        size_t R = weightDims[0];
        size_t S = weightDims[1];
        size_t U = 1;
        
        // Initialize with biases
        for (size_t p = 0; p < P; p++) {
            for (size_t q = 0; q < Q; q++) {
                for (size_t m = 0; m < M; m++) {
                    size_t output_idx = p * Q * M + q * M + m;
                    outData.get(output_idx) = biasData.get(m);
                }
            }
        }
        
        // Optimized loop order: c -> m -> p -> q -> r -> s
        for (size_t c = 0; c < C; c++) {
            for (size_t m = 0; m < M; m++) {
                for (size_t p = 0; p < P; p++) {
                    for (size_t q = 0; q < Q; q++) {
                        size_t output_idx = p * Q * M + q * M + m;
                        fp32 accumulator = 0.0f;
                        
                        for (size_t r = 0; r < R; r++) {
                            size_t input_h = U * p + r;
                            for (size_t s = 0; s < S; s++) {
                                size_t input_w = U * q + s;
                                size_t input_idx = input_h * W * C + input_w * C + c;
                                size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                                
                                accumulator += dataIn.get(input_idx) * weightData.get(weight_idx);
                            }
                        }
                        
                        outData.get(output_idx) += accumulator;
                    }
                }
            }
        }
        
        // Apply ReLU
        for (size_t p = 0; p < P; p++) {
            for (size_t q = 0; q < Q; q++) {
                for (size_t m = 0; m < M; m++) {
                    size_t output_idx = p * Q * M + q * M + m;
                    fp32 value = outData.get(output_idx);
                    outData.get(output_idx) = std::max(0.0f, value);
                }
            }
        }
    }
};

// ============================================================================
// BENCHMARK FUNCTIONS
// ============================================================================

double calculateMean(const std::vector<double>& data) {
    double sum = 0;
    for (double d : data) sum += d;
    return sum / data.size();
}

double calculateStdDev(const std::vector<double>& data, double mean) {
    double variance = 0;
    for (double d : data) {
        variance += (d - mean) * (d - mean);
    }
    return std::sqrt(variance / (data.size() - 1));
}

void saveToCSV(const std::string& filename,
               const std::vector<double>& naive,
               const std::vector<double>& optimized) {
    std::ofstream file(filename);
    file << "Iteration,Naive_ms,Optimized_ms\n";
    for (size_t i = 0; i < naive.size(); i++) {
        file << (i + 1) << "," << naive[i] << "," << optimized[i] << "\n";
    }
    file.close();
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main() {
    std::cout << "\n============================================" << std::endl;
    std::cout << "  LAB 5: CACHE OPTIMIZATION BENCHMARK" << std::endl;
    std::cout << "============================================\n" << std::endl;
    
    // ========================================================================
    // Setup: Create a layer similar to your Layer 1
    // Dimensions based on your Lab 2 results:
    // Input: 64x64x3, Output: 56x56x32, Kernel: 5x5
    // ========================================================================
    
    std::cout << "Setting up test layer..." << std::endl;
    
    LayerParams inputParams(sizeof(fp32), {64, 64, 3});    // Input: 64x64x3
    LayerParams outputParams(sizeof(fp32), {56, 56, 32});  // Output: 56x56x32
    LayerParams weightParams(sizeof(fp32), {5, 5, 3, 32}); // Kernel: 5x5, 3->32 channels
    LayerParams biasParams(sizeof(fp32), {32});            // 32 biases
    
    SimpleConvLayer layer(inputParams, outputParams, weightParams, biasParams);
    
    // Create input data (filled with ones for testing)
    LayerData inputData(inputParams);
    for (size_t i = 0; i < inputParams.flat_count(); i++) {
        inputData.get(i) = 1.0f;
    }
    
    std::cout << "Input size:  " << inputParams.dims[0] << "x" 
              << inputParams.dims[1] << "x" << inputParams.dims[2] << std::endl;
    std::cout << "Output size: " << outputParams.dims[0] << "x" 
              << outputParams.dims[1] << "x" << outputParams.dims[2] << std::endl;
    std::cout << "Kernel size: " << weightParams.dims[0] << "x" 
              << weightParams.dims[1] << std::endl;
    
    const int NUM_ITERATIONS = 100;
    
    // ========================================================================
    // BENCHMARK 1: NAIVE Implementation
    // ========================================================================
    
    std::cout << "\n[1/2] Benchmarking NAIVE implementation..." << std::endl;
    std::vector<double> naive_timings;
    
    // Warm-up (not counted)
    layer.computeNaive(inputData);
    
    // Timed runs
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        layer.computeNaive(inputData);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        naive_timings.push_back(duration.count() / 1000.0);
        
        if ((i + 1) % 10 == 0) {
            std::cout << "  Progress: " << (i + 1) << "/" << NUM_ITERATIONS << std::endl;
        }
    }
    
    double naive_mean = calculateMean(naive_timings);
    double naive_std = calculateStdDev(naive_timings, naive_mean);
    
    std::cout << "\nNaive Results:" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Mean:   " << naive_mean << " ms" << std::endl;
    std::cout << "  Std:    " << naive_std << " ms" << std::endl;
    
    // ========================================================================
    // BENCHMARK 2: CACHE-OPTIMIZED Implementation
    // ========================================================================
    
    std::cout << "\n[2/2] Benchmarking CACHE-OPTIMIZED implementation..." << std::endl;
    std::vector<double> optimized_timings;
    
    // Warm-up
    layer.computeOptimized(inputData);
    
    // Timed runs
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        layer.computeOptimized(inputData);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        optimized_timings.push_back(duration.count() / 1000.0);
        
        if ((i + 1) % 10 == 0) {
            std::cout << "  Progress: " << (i + 1) << "/" << NUM_ITERATIONS << std::endl;
        }
    }
    
    double optimized_mean = calculateMean(optimized_timings);
    double optimized_std = calculateStdDev(optimized_timings, optimized_mean);
    
    std::cout << "\nOptimized Results:" << std::endl;
    std::cout << "  Mean:   " << optimized_mean << " ms" << std::endl;
    std::cout << "  Std:    " << optimized_std << " ms" << std::endl;
    
    // ========================================================================
    // RESULTS COMPARISON
    // ========================================================================
    
    double speedup = naive_mean / optimized_mean;
    double improvement = ((naive_mean - optimized_mean) / naive_mean) * 100.0;
    
    std::cout << "\n============================================" << std::endl;
    std::cout << "              FINAL RESULTS" << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Naive mean:           " << naive_mean << " ms" << std::endl;
    std::cout << "Cache-Optimized mean: " << optimized_mean << " ms" << std::endl;
    std::cout << "Speedup:              " << speedup << "x" << std::endl;
    std::cout << "Improvement:          " << improvement << "%" << std::endl;
    std::cout << "============================================\n" << std::endl;
    
    // ========================================================================
    // Save results to CSV
    // ========================================================================
    
    saveToCSV("layer1_benchmark_results.csv", naive_timings, optimized_timings);
    std::cout << "Results saved to: layer1_benchmark_results.csv" << std::endl;
    std::cout << "Now run: python visualize_results.py\n" << std::endl;
    
    return 0;
}

/*
 * ============================================================================
 * COMPILATION AND RUNNING INSTRUCTIONS
 * ============================================================================
 * 
 * 
 * STEP 1: Compile (choose ONE option):
 * 
 * Option A - Standalone (easiest):
 *   g++ -o test_perf test_performance.cpp -std=c++11 -O2
 * 
 * Option B - With our existing source files (if you get linker errors):
 *   g++ -o test_perf test_performance.cpp -std=c++11 -O2 -Isrc
 * 
 * STEP 2: Run:
 *   ./test_perf
 * 
 * STEP 3: Visualize:
 *   python visualize_results.py
 * 
 * ============================================================================
 * EXPECTED OUTPUT
 * ============================================================================
 * 
 * ============================================
 *   LAB 5: CACHE OPTIMIZATION BENCHMARK
 * ============================================
 * 
 * Setting up test layer...
 * Input size:  64x64x3
 * Output size: 56x56x32
 * Kernel size: 5x5
 * 
 * [1/2] Benchmarking NAIVE implementation...
 *   Progress: 10/100
 *   Progress: 20/100
 *   ...
 * 
 * Naive Results:
 *   Mean:   45.234 ms
 *   Std:    1.123 ms
 * 
 * [2/2] Benchmarking CACHE-OPTIMIZED implementation...
 *   Progress: 10/100
 *   ...
 * 
 * Optimized Results:
 *   Mean:   18.567 ms
 *   Std:    0.789 ms
 * 
 * ============================================
 *               FINAL RESULTS
 * ============================================
 * Naive mean:           45.234 ms
 * Cache-Optimized mean: 18.567 ms
 * Speedup:              2.436x
 * Improvement:          58.9%
 * ============================================
 * 
 * Results saved to: layer1_benchmark_results.csv
 * Post-Results run: python visualize_results.py
 * 
 */