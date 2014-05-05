#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

// QRNG init kernel
__global__ void init_QRNG(curandStateSobol32_t *const rngStates,
                          curandDirectionVectors32_t *const rngDirections)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    curand_init(rngDirections[0], 2*tid, &rngStates[tid]);
    curand_init(rngDirections[1], 2*tid, &rngStates[tid + step]);
}

__global__ void generate_uniforms(float *const results,
                                  curandStateSobol32_t *const rngStates,
                                  const unsigned int N)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int step = gridDim.x * blockDim.x;

    // Copy state to local memory for efficiency.
    curandStateSobol32_t localState1 = rngStates[tid];
    curandStateSobol32_t localState2 = rngStates[tid + step];

    // Generate up to N random points and save each one.
    for (unsigned int i=tid ; i<N ; i+=step)
    {
        float x = curand_uniform(&localState1);
        float y = curand_uniform(&localState2);

        results[2*i+0] = x;
        results[2*i+1] = y;
    }
}

int main(void) {

    dim3 block;
    dim3 grid;

    // The number of random points we wish to generate.
    unsigned int N = 32;

    // A maximum/optimum for our hypothetical device.
    unsigned int threadBlockSize = 4;

    block.x = threadBlockSize;
    // Grid size required to cover all work with unique threads.
    grid.x  = (N + threadBlockSize - 1) / threadBlockSize;

    // Reduced grid size that fits our hypothetical device.
    grid.x = 2;

    // Allocate memory for RNG states and direction vectors.
    // Factor of two since each thread generates independent x and y values.
    curandStateSobol32_t       *d_rngStates     = 0;
    curandDirectionVectors32_t *d_rngDirections = 0;
    cudaMalloc((void **)&d_rngStates,
               2 * grid.x * block.x * sizeof(curandStateSobol32_t));
    cudaMalloc((void **)&d_rngDirections,
               2 * sizeof(curandDirectionVectors32_t));

    // Allocate memory for result
    float *d_results = 0;
    cudaMalloc((void **)&d_results, 2 * N * sizeof(float));

    // Generate direction vectors on the host and copy to the device.
    curandDirectionVectors32_t *rngDirections;
    curandGetDirectionVectors32(&rngDirections,
                                CURAND_DIRECTION_VECTORS_32_JOEKUO6);
    cudaMemcpy(d_rngDirections, rngDirections,
               2 * sizeof(curandDirectionVectors32_t), cudaMemcpyHostToDevice);

    // Generate N points from 2*N quasirandom numbers.
    init_QRNG<<<grid, block>>>(d_rngStates, d_rngDirections);
    generate_uniforms<<<grid, block>>>(d_results, d_rngStates, N);

    // Allocate memory on host for floats and copy from device.
    float* h_results = new float[2*N];
    cudaMemcpy(h_results, d_results,
               2 * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results.
    // Note that lines 9-15 of the output are idential to lines 2-8!
    std::cout.setf(std::ios::fixed, std::ios::floatfield);
    std::cout.precision(5);
    for (unsigned int i=0; i<N; i++) {
        std::cout << "x: " << h_results[2*i+0] << ", y: " << h_results[2*i+1]
                  << std::endl;
    }

    cudaFree(d_rngStates);
    cudaFree(d_rngDirections);
    cudaFree(d_results);
    delete [] h_results;

    return 0;
}
