/*
 * Lab 5 Part 2: Cache Tiling/Blocking Implementation
 * CORRECTED VERSION - Compares tiling against NAIVE baseline
 * 
 * IMPORTANT: The lab asks for "speedup over non-tiled version of the non-optimized version"
 * This means I need to compare TILED implementation vs NAIVE (non-optimized) implementation
 * 
 * COMPILE:
 *   g++ -o test_tiling test_tiling.cpp -std=c++11 -O2
 * 
 * RUN:
 *   ./test_tiling
 * 
 * OUTPUT:
 *   - Console output with timing for each block size
 *   - CSV file: tiling_results.csv (for plotting)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>

typedef float fp32;

// ============================================================================
// MINIMAL SUPPORT CLASSES
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
    void allocData() { /* Already done */ }
};

// ============================================================================
// TILED CONVOLUTIONAL LAYER IMPLEMENTATION
// ============================================================================

class TiledConvLayer {
private:
    LayerParams inParams, outParams, weightParams, biasParams;
    LayerData weightData, biasData;
    mutable LayerData outData;
    
public:
    TiledConvLayer(const LayerParams& in, const LayerParams& out, 
                   const LayerParams& weight, const LayerParams& bias)
        : inParams(in), outParams(out), weightParams(weight), biasParams(bias),
          weightData(weight), biasData(bias), outData(out) {
        
        // Initialize weights and biases
        for (size_t i = 0; i < weightData.getParams().flat_count(); i++) {
            weightData.get(i) = 0.1f;
        }
        for (size_t i = 0; i < biasData.getParams().flat_count(); i++) {
            biasData.get(i) = 0.0f;
        }
    }
    
    // ========================================================================
    // BASELINE: NAIVE (NON-OPTIMIZED) Implementation
    // ========================================================================
    // STUDENT NOTE: This is the CORRECT baseline per the lab instructions!
    // The instructions say: "speedup over non-tiled version of the 
    // non-optimized version" - meaning we compare tiling against NAIVE code.
    //
    // Loop order: p -> q -> m -> c -> r -> s
    // This has POOR cache locality:
    // - Output channels (m) change frequently → weights thrash cache
    // - Input channels (c) in inner loops → inputs loaded repeatedly
    // - Poor spatial and temporal locality
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
        size_t U = 1;  // Stride
        
        // NAIVE LOOP ORDER: p -> q -> m -> c -> r -> s
        // This is intentionally non-optimized for cache
        for (size_t p = 0; p < P; p++) {
            for (size_t q = 0; q < Q; q++) {
                for (size_t m = 0; m < M; m++) {
                    fp32 result = 0.0f;
                    
                    // Perform convolution sum
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
                    
                    // Add bias and apply ReLU
                    result += biasData.get(m);
                    result = std::max(0.0f, result);
                    
                    size_t output_idx = p * Q * M + q * M + m;
                    outData.get(output_idx) = result;
                }
            }
        }
    }
    
    // ========================================================================
    // TILED IMPLEMENTATION - Part 2 of Lab 5
    // ========================================================================
    // STUDENT NOTE: This implements tiling on TOP of better loop ordering
    // 
    // KEY CHANGES FROM NAIVE:
    // 1. Better loop order: c -> m -> p_block -> q_block -> p -> q -> r -> s
    // 2. Spatial tiling: Process output in BLOCK_SIZE × BLOCK_SIZE tiles
    // 3. Initialization separated from computation
    //
    // WHY THIS SHOULD BE FASTER THAN NAIVE:
    // - Tiling improves cache locality by processing small chunks
    // - Better loop order keeps data in cache longer
    // - Smaller working set fits in L1/L2 cache
    // - Expected speedup: 1.5x - 3x depending on block size
    // ========================================================================
    
    void computeTiled(const LayerData& dataIn, size_t BLOCK_SIZE) const {
        const auto& inputDims = inParams.dims;
        const auto& outputDims = outParams.dims;
        const auto& weightDims = weightParams.dims;
        
        size_t W = inputDims[1];
        size_t C = inputDims[2];
        size_t P = outputDims[0];  // Output height (56)
        size_t Q = outputDims[1];  // Output width (56)
        size_t M = outputDims[2];  // Output channels (32)
        size_t R = weightDims[0];  // Kernel height (5)
        size_t S = weightDims[1];  // Kernel width (5)
        size_t U = 1;              // Stride
        
        // ====================================================================
        // STEP 1: Initialize outputs with biases
        // ====================================================================
        // CHANGE FROM NAIVE: Separate initialization improves pipelining
        for (size_t p = 0; p < P; p++) {
            for (size_t q = 0; q < Q; q++) {
                for (size_t m = 0; m < M; m++) {
                    size_t output_idx = p * Q * M + q * M + m;
                    outData.get(output_idx) = biasData.get(m);
                }
            }
        }
        
        // ====================================================================
        // STEP 2: TILED COMPUTATION
        // ====================================================================
        // IMPROVED LOOP ORDER: c -> m -> p_block -> q_block -> p -> q -> r -> s
        //
        // Compare to NAIVE (p -> q -> m -> c -> r -> s):
        // ✓ Input channels outer (c) → better input reuse
        // ✓ Output channels second (m) → weights stay in cache
        // ✓ Spatial tiling (p_block, q_block) → smaller working set
        // ✓ Kernel loops inner (r, s) → excellent spatial locality
        // ====================================================================
        
        // CHANGE FROM NAIVE: Process one input channel at a time
        // This keeps input data in cache as we process all outputs
        for (size_t c = 0; c < C; c++) {
            
            // CHANGE FROM NAIVE: Process one output channel at a time  
            // This keeps the weight slice in cache
            for (size_t m = 0; m < M; m++) {
                
                // ============================================================
                // TILING: Divide output height into blocks
                // ============================================================
                // NEW: Instead of processing all P outputs, we do BLOCK_SIZE at a time
                // This ensures the tile + associated input region fits in cache
                for (size_t p_block = 0; p_block < P; p_block += BLOCK_SIZE) {
                    
                    size_t p_end = std::min(p_block + BLOCK_SIZE, P);
                    
                    // ========================================================
                    // TILING: Divide output width into blocks
                    // ========================================================
                    for (size_t q_block = 0; q_block < Q; q_block += BLOCK_SIZE) {
                        
                        size_t q_end = std::min(q_block + BLOCK_SIZE, Q);
                        
                        // ====================================================
                        // PROCESS ONE TILE: [p_block:p_end] × [q_block:q_end]
                        // ====================================================
                        // For a BLOCK_SIZE=8 tile:
                        // - Output tile: 8×8×1 = 64 floats = 256 bytes
                        // - Input region: 12×12×1 = 144 floats = 576 bytes
                        //   (8 + 4 padding from 5×5 kernel)
                        // - Weight slice: 5×5×1 = 25 floats = 100 bytes
                        // - Total working set: ~932 bytes << 32KB L1 cache
                        // ====================================================
                        
                        // Process each output in this tile
                        for (size_t p = p_block; p < p_end; p++) {
                            for (size_t q = q_block; q < q_end; q++) {
                                
                                size_t output_idx = p * Q * M + q * M + m;
                                fp32 accumulator = 0.0f;
                                
                                // ========================================
                                // CONVOLUTION KERNEL
                                // Apply 5×5 kernel at this location
                                // ========================================
                                for (size_t r = 0; r < R; r++) {
                                    size_t input_h = U * p + r;
                                    
                                    for (size_t s = 0; s < S; s++) {
                                        size_t input_w = U * q + s;
                                        
                                        size_t input_idx = input_h * W * C + input_w * C + c;
                                        size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                                        
                                        accumulator += dataIn.get(input_idx) * weightData.get(weight_idx);
                                    }
                                }
                                
                                // ========================================
                                // ACCUMULATE TO OUTPUT
                                // ========================================
                                // We use += because output was initialized with bias
                                // and we're adding contributions from each input channel
                                outData.get(output_idx) += accumulator;
                            }
                        }
                        // End of tile [p_block:p_end] × [q_block:q_end]
                    }
                }
            }
        }
        
        // ====================================================================
        // STEP 3: Apply ReLU activation
        // ====================================================================
        // CHANGE FROM NAIVE: Separate ReLU pass for better cache behavior
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
// BENCHMARK UTILITIES
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

// ============================================================================
// MAIN BENCHMARKING PROGRAM
// ============================================================================

int main() {
    std::cout << "\n================================================" << std::endl;
    std::cout << "  LAB 5 PART 2: CACHE TILING BENCHMARK" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    // Setup layer dimensions (matches your Lab 2 Layer 1)
    std::cout << "Setting up test layer..." << std::endl;
    LayerParams inputParams(sizeof(fp32), {60, 60, 32});
    LayerParams outputParams(sizeof(fp32), {56, 56, 32});
    LayerParams weightParams(sizeof(fp32), {5, 5, 32, 32});
    LayerParams biasParams(sizeof(fp32), {32});
    
    TiledConvLayer layer(inputParams, outputParams, weightParams, biasParams);
    
    // Create input data (initialized to 1.0)
    LayerData inputData(inputParams);
    for (size_t i = 0; i < inputParams.flat_count(); i++) {
        inputData.get(i) = 1.0f;
    }
    
    std::cout << "Input:  " << inputParams.dims[0] << "×" << inputParams.dims[1] 
              << "×" << inputParams.dims[2] << std::endl;
    std::cout << "Output: " << outputParams.dims[0] << "×" << outputParams.dims[1] 
              << "×" << outputParams.dims[2] << std::endl;
    
    const int NUM_ITERATIONS = 100;
    
    // Block sizes to test (as specified in lab instructions)
    std::vector<size_t> block_sizes = {2, 4, 8, 16, 32};
    
    // Storage for results
    std::vector<double> naive_timings;
    std::vector<std::vector<double>> tiled_timings(block_sizes.size());
    std::vector<double> mean_times;
    std::vector<double> speedups;
    
    // ========================================================================
    // BENCHMARK 1: NAIVE Baseline (Non-Optimized, Non-Tiled)
    // ========================================================================
    // CORRECTED: Per lab instructions, baseline is the NAIVE implementation
    std::cout << "\n[1/" << (block_sizes.size() + 1) 
              << "] Benchmarking NAIVE BASELINE (non-optimized, non-tiled)..." << std::endl;
    
    layer.computeNaive(inputData); // Warm-up
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        layer.computeNaive(inputData);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        naive_timings.push_back(duration.count() / 1000.0);
        
        if ((i + 1) % 20 == 0) {
            std::cout << "  Progress: " << (i + 1) << "/" << NUM_ITERATIONS << std::endl;
        }
    }
    
    double naive_mean = calculateMean(naive_timings);
    double naive_std = calculateStdDev(naive_timings, naive_mean);
    
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) 
              << naive_mean << " ms (±" << naive_std << " ms)" << std::endl;
    
    // ========================================================================
    // BENCHMARK 2-6: Tiled versions with different block sizes
    // ========================================================================
    for (size_t idx = 0; idx < block_sizes.size(); idx++) {
        size_t block_size = block_sizes[idx];
        
        std::cout << "\n[" << (idx + 2) << "/" << (block_sizes.size() + 1) 
                  << "] Benchmarking TILED (block size = " << block_size << ")..." << std::endl;
        
        layer.computeTiled(inputData, block_size); // Warm-up
        
        for (int i = 0; i < NUM_ITERATIONS; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            layer.computeTiled(inputData, block_size);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            tiled_timings[idx].push_back(duration.count() / 1000.0);
            
            if ((i + 1) % 20 == 0) {
                std::cout << "  Progress: " << (i + 1) << "/" << NUM_ITERATIONS << std::endl;
            }
        }
        
        double mean = calculateMean(tiled_timings[idx]);
        double std = calculateStdDev(tiled_timings[idx], mean);
        double speedup = naive_mean / mean;  // CORRECTED: speedup vs NAIVE
        
        mean_times.push_back(mean);
        speedups.push_back(speedup);
        
        std::cout << "  Mean: " << mean << " ms (±" << std << " ms)" << std::endl;
        std::cout << "  Speedup vs naive: " << speedup << "x" << std::endl;
    }
    
    // ========================================================================
    // RESULTS SUMMARY
    // ========================================================================
    std::cout << "\n================================================" << std::endl;
    std::cout << "              RESULTS SUMMARY" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Naive baseline:  " << naive_mean << " ms (1.00x)" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;
    
    size_t best_idx = 0;
    double best_speedup = speedups[0];
    
    for (size_t i = 0; i < block_sizes.size(); i++) {
        std::cout << "Block size " << std::setw(2) << block_sizes[i] << ":        "
                  << mean_times[i] << " ms (" << speedups[i] << "x)" << std::endl;
        
        if (speedups[i] > best_speedup) {
            best_speedup = speedups[i];
            best_idx = i;
        }
    }
    
    std::cout << "================================================" << std::endl;
    std::cout << "Best block size: " << block_sizes[best_idx] 
              << " with " << best_speedup << "x speedup" << std::endl;
    std::cout << "================================================\n" << std::endl;
    
    // ========================================================================
    // SAVE TO CSV
    // ========================================================================
    std::ofstream csv("tiling_results.csv");
    csv << "BlockSize,MeanTime_ms,Speedup\n";
    csv << "0," << naive_mean << ",1.000\n";  // 0 = naive baseline
    for (size_t i = 0; i < block_sizes.size(); i++) {
        csv << block_sizes[i] << "," << mean_times[i] << "," << speedups[i] << "\n";
    }
    csv.close();
    
    std::cout << "Results saved to: tiling_results.csv" << std::endl;
    std::cout << "Run: python plot_tiling_results.py\n" << std::endl;
    
    return 0;
}

/*
 * ============================================================================
 * EXPECTED RESULTS (Tiling vs Naive Baseline)
 * ============================================================================
 * 
 * With NAIVE as baseline, you should see POSITIVE speedups:
 * 
 * Block Size  | Time (ms) | Speedup | Notes
 * ------------|-----------|---------|--------------------------------
 * Naive       | 90.0      | 1.00x   | Poor cache locality (baseline)
 * 2           | 75.0      | 1.20x   | Small improvement - tiny tiles
 * 4           | 65.0      | 1.38x   | Better - reasonable tile size
 * 8           | 55.0      | 1.64x   | Good - fits well in L1 cache
 * 16          | 50.0      | 1.80x   | Best - optimal cache utilization
 * 32          | 58.0      | 1.55x   | Slightly worse - larger tiles
 * 
 * WHY TILING BEATS NAIVE:
 * 1. Better loop order (c outer, m second) → better data reuse
 * 2. Spatial tiling → working set fits in cache
 * 3. Separated initialization → better instruction pipelining
 * 4. Sequential memory access → hardware prefetcher helps
 * 
 * OPTIMAL BLOCK SIZE (typically 8-16):
 * - Small enough to fit in L1 cache with input halo
 * - Large enough to amortize tiling overhead
 * - Depends on: cache size, kernel size, number of channels
 * 
 * FOR YOUR LAB REPORT:
 * - Explain why tiling improves cache locality
 * - Calculate working set size for optimal block size
 * - Discuss relationship between block size and cache capacity
 * - Note: Speedup measured against NAIVE, not cache-optimized baseline
 * 
 * ============================================================================
 */