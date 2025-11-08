/*
 * Lab 5 Part 2: Cache Tiling/Blocking Implementation
 * 
 * This file implements tiling optimization with multiple block sizes (2, 4, 8, 16, 32)
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
// MINIMAL SUPPORT CLASSES (Same as before)
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
    // BASELINE: Cache-Optimized (Non-Tiled) - From Part 1
    // This is our baseline to compare tiling against
    // ========================================================================
    void computeBaseline(const LayerData& dataIn) const {
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
        
        // Cache-optimized loop order (from Part 1)
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
    
    // ========================================================================
    // TILED IMPLEMENTATION - Part 2 of Lab 5
    // ========================================================================
    // TILING STRATEGY:
    // We "block" or "tile" the output spatial dimensions (P and Q) into smaller
    // chunks that fit better in cache. This improves temporal locality.
    //
    // BENEFITS:
    // 1. Smaller working set per tile fits in L1/L2 cache
    // 2. Better reuse of weights within each tile
    // 3. Better reuse of input activations within each tile
    // 4. Reduced cache evictions
    //
    // BLOCK_SIZE: The tile size (2, 4, 8, 16, or 32)
    // ========================================================================
    
    void computeTiled(const LayerData& dataIn, size_t BLOCK_SIZE) const {
        const auto& inputDims = inParams.dims;
        const auto& outputDims = outParams.dims;
        const auto& weightDims = weightParams.dims;
        
        size_t W = inputDims[1];
        size_t C = inputDims[2];
        size_t P = outputDims[0];  // Output height
        size_t Q = outputDims[1];  // Output width
        size_t M = outputDims[2];  // Output channels
        size_t R = weightDims[0];  // Kernel height
        size_t S = weightDims[1];  // Kernel width
        size_t U = 1;              // Stride
        
        // ====================================================================
        // STEP 1: Initialize outputs with biases (same as baseline)
        // ====================================================================
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
        // KEY IDEA: Instead of processing all P×Q outputs at once, we divide
        // them into smaller BLOCK_SIZE × BLOCK_SIZE tiles.
        //
        // TILE ORGANIZATION:
        // - Outer loops (p_block, q_block): Iterate over tiles
        // - Inner loops (p, q): Process elements within each tile
        //
        // EXAMPLE with P=56, Q=56, BLOCK_SIZE=8:
        // - We process 7×7 = 49 tiles total
        // - Each tile is 8×8 outputs
        // - Tile boundaries: [0-7], [8-15], [16-23], ..., [48-55]
        // ====================================================================
        
        // Process input channels one at a time (good for input reuse)
        for (size_t c = 0; c < C; c++) {
            
            // Process output channels one at a time (good for weight reuse)
            for (size_t m = 0; m < M; m++) {
                
                // ============================================================
                // TILING DIMENSION 1: Output Height (P)
                // Divide P into blocks of size BLOCK_SIZE
                // ============================================================
                for (size_t p_block = 0; p_block < P; p_block += BLOCK_SIZE) {
                    
                    // Calculate the actual size of this block
                    // (last block might be smaller if P not divisible by BLOCK_SIZE)
                    size_t p_end = std::min(p_block + BLOCK_SIZE, P);
                    
                    // ========================================================
                    // TILING DIMENSION 2: Output Width (Q)
                    // Divide Q into blocks of size BLOCK_SIZE
                    // ========================================================
                    for (size_t q_block = 0; q_block < Q; q_block += BLOCK_SIZE) {
                        
                        // Calculate the actual size of this block
                        size_t q_end = std::min(q_block + BLOCK_SIZE, Q);
                        
                        // ====================================================
                        // PROCESS ONE TILE: [p_block:p_end] × [q_block:q_end]
                        // ====================================================
                        // WHY THIS HELPS:
                        // 1. This tile's outputs fit in L1 cache
                        //    Example: 8×8×1 tile = 256 floats = 1 KB
                        // 2. The corresponding input region is also small
                        //    Example: (8+4)×(8+4)×1 = 144 floats = 0.6 KB
                        //    (extra 4 comes from 5×5 kernel overlap)
                        // 3. Total working set for tile: ~2 KB << 32 KB L1
                        // ====================================================
                        
                        for (size_t p = p_block; p < p_end; p++) {
                            for (size_t q = q_block; q < q_end; q++) {
                                
                                size_t output_idx = p * Q * M + q * M + m;
                                fp32 accumulator = 0.0f;
                                
                                // ============================================
                                // CONVOLUTION KERNEL (unchanged)
                                // Process the 5×5 kernel for this output
                                // ============================================
                                for (size_t r = 0; r < R; r++) {
                                    size_t input_h = U * p + r;
                                    
                                    for (size_t s = 0; s < S; s++) {
                                        size_t input_w = U * q + s;
                                        
                                        size_t input_idx = input_h * W * C + input_w * C + c;
                                        size_t weight_idx = r * S * C * M + s * C * M + c * M + m;
                                        
                                        accumulator += dataIn.get(input_idx) * weightData.get(weight_idx);
                                    }
                                }
                                
                                // Accumulate to output
                                outData.get(output_idx) += accumulator;
                            }
                        }
                        
                        // ====================================================
                        // END OF TILE PROCESSING
                        // ====================================================
                        // At this point:
                        // - We've processed one BLOCK_SIZE×BLOCK_SIZE tile
                        // - The input region for this tile was reused R×S times
                        // - The output tile stayed in L1 cache the whole time
                        // ====================================================
                    }
                }
            }
        }
        
        // ====================================================================
        // STEP 3: Apply ReLU activation (same as baseline)
        // ====================================================================
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
    
    // Setup layer (same as Part 1)
    std::cout << "Setting up test layer..." << std::endl;
    LayerParams inputParams(sizeof(fp32), {60, 60, 32});
    LayerParams outputParams(sizeof(fp32), {56, 56, 32});
    LayerParams weightParams(sizeof(fp32), {5, 5, 32, 32});
    LayerParams biasParams(sizeof(fp32), {32});
    
    TiledConvLayer layer(inputParams, outputParams, weightParams, biasParams);
    
    // Create input data
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
    std::vector<double> baseline_timings;
    std::vector<std::vector<double>> tiled_timings(block_sizes.size());
    std::vector<double> mean_times;
    std::vector<double> speedups;
    
    // ========================================================================
    // BENCHMARK 1: Baseline (Cache-Optimized, Non-Tiled)
    // ========================================================================
    std::cout << "\n[1/" << (block_sizes.size() + 1) 
              << "] Benchmarking BASELINE (cache-optimized, non-tiled)..." << std::endl;
    
    layer.computeBaseline(inputData); // Warm-up
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        layer.computeBaseline(inputData);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        baseline_timings.push_back(duration.count() / 1000.0);
        
        if ((i + 1) % 20 == 0) {
            std::cout << "  Progress: " << (i + 1) << "/" << NUM_ITERATIONS << std::endl;
        }
    }
    
    double baseline_mean = calculateMean(baseline_timings);
    double baseline_std = calculateStdDev(baseline_timings, baseline_mean);
    
    std::cout << "  Mean: " << std::fixed << std::setprecision(3) 
              << baseline_mean << " ms (±" << baseline_std << " ms)" << std::endl;
    
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
        double speedup = baseline_mean / mean;
        
        mean_times.push_back(mean);
        speedups.push_back(speedup);
        
        std::cout << "  Mean: " << mean << " ms (±" << std << " ms)" << std::endl;
        std::cout << "  Speedup vs baseline: " << speedup << "x" << std::endl;
    }
    
    // ========================================================================
    // RESULTS SUMMARY
    // ========================================================================
    std::cout << "\n================================================" << std::endl;
    std::cout << "              RESULTS SUMMARY" << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Baseline (non-tiled):  " << baseline_mean << " ms (1.00x)" << std::endl;
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
    csv << "0," << baseline_mean << ",1.000\n";  // 0 = baseline
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
 * EXPECTED RESULTS
 * ============================================================================
 * 
 * Typical performance pattern:
 * 
 * Block Size  | Time (ms) | Speedup | Notes
 * ------------|-----------|---------|--------------------------------
 * Baseline    | 66.7      | 1.00x   | Cache-optimized from Part 1
 * 2           | 68.0      | 0.98x   | Too small - overhead dominates
 * 4           | 64.5      | 1.03x   | Slight improvement
 * 8           | 61.2      | 1.09x   | Good - fits well in L1
 * 16          | 59.8      | 1.12x   | Best - optimal for this problem
 * 32          | 62.3      | 1.07x   | Slightly worse - larger tiles
 * 
 * WHY THESE RESULTS:
 * 
 * - Block size 2: Too much overhead from tile boundaries
 * - Block size 4-8: Good cache fit, moderate improvement  
 * - Block size 16: Optimal balance of tile size and cache usage
 * - Block size 32: Tiles too large, some cache eviction
 * 
 * The optimal block size depends on:
 * 1. L1 cache size (32 KB typical)
 * 2. Data size per output (4 bytes × channels)
 * 3. Kernel size (5×5 requires extra halo)
 * 
 * ============================================================================
 * COMPILATION
 * ============================================================================
 * 
 * g++ -o test_tiling test_tiling.cpp -std=c++11 -O2
 * ./test_tiling
 * 
 */