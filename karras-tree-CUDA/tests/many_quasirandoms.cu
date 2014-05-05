#include <curand_kernel.h>

#include <iostream>
#include <fstream>

inline void cuda_call(cudaError_t code, char* file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error!\nMsg:  %s\nFile: %s @ line %d\n",
                cudaGetErrorString(code), file, line);

    if (abort)
        exit(code);
    }
}

__global__ void init_QRNG_no_loop(curandStateSobol32_t *const qrng_states,
                                  curandDirectionVectors32_t *const qrng_directions,
                                  const unsigned int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialise the Q-RNG
    // max(tid) >= N, so we generate N states.
    if (tid < N) {
        curand_init(qrng_directions[0], tid+2, &qrng_states[tid+0*N]); // x
        curand_init(qrng_directions[1], tid+2, &qrng_states[tid+1*N]); // y
        curand_init(qrng_directions[2], tid+2, &qrng_states[tid+2*N]); // z
    }
}

__global__ void init_QRNG_reuse_states(curandStateSobol32_t *const qrng_states,
                                       curandDirectionVectors32_t *const qrng_directions,
                                       const unsigned int N_per_thread)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Initialise the Q-RNG
    // We may have that max(tid) < N, and only generate max(tid) states. States
    // are reused. Each state's initializing offset is such that it will not
    // 'catch up' to the state proceeding it when we generate N numbers.

    curand_init(qrng_directions[0], (N_per_thread*tid)+2, &qrng_states[tid+0*stride]); // x
    curand_init(qrng_directions[1], (N_per_thread*tid)+2, &qrng_states[tid+1*stride]); // y
    curand_init(qrng_directions[2], (N_per_thread*tid)+2, &qrng_states[tid+2*stride]); // z
}


__global__ void gen_normals_no_loop(float *normals,
                                    curandStateSobol32_t *const qrng_states,
                                    const unsigned int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < N) {
        curandStateSobol32_t x_state = qrng_states[tid+0*N];
        curandStateSobol32_t y_state = qrng_states[tid+1*N];
        curandStateSobol32_t z_state = qrng_states[tid+2*N];

        float x, y, z;

        x = curand_normal(&x_state);
        y = curand_normal(&y_state);
        z = curand_normal(&z_state);

        normals[tid+0*N] = x;
        normals[tid+1*N] = y;
        normals[tid+2*N] = z;
    }
}

__global__ void gen_normals_reuse_states(float *normals,
                                         curandStateSobol32_t *const qrng_states,
                                         const size_t N,
                                         const unsigned int N_per_thread)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Output starting index.
    unsigned int oid = N_per_thread * tid;
    unsigned int stride = blockDim.x * gridDim.x;

    curandStateSobol32_t x_state = qrng_states[tid+0*stride];
    curandStateSobol32_t y_state = qrng_states[tid+1*stride];
    curandStateSobol32_t z_state = qrng_states[tid+2*stride];

    float x, y, z;

    while (tid < N)
    {
        x = curand_normal(&x_state);
        y = curand_normal(&y_state);
        z = curand_normal(&z_state);

        // Note that we output in the same order as the above kernel.
        normals[oid+0*N] = x;
        normals[oid+1*N] = y;
        normals[oid+2*N] = z;

        tid += stride;
        oid++;
    }
}

int main(void) {

    /* Data common to both methods. */

    // Set up for a limited number of threads.
    dim3 grid, block;
    grid.x = 7;
    block.x = 512;

    unsigned int N_per_thread = 10;

    // Set such that each state-reusing thread calls curand_normal exactly
    // N_per_thread times.
    const size_t N = N_per_thread * grid.x * block.x;
    std::cout << "N:            " << N << std::endl;
    std::cout << "N per thread: " << N_per_thread << std::endl;
    std::cout << std::endl;

    curandStateSobol32_t *d_qrng_states;

    // Generate Q-RNG 'direction vectors' on host, and copy to device.
    curandDirectionVectors32_t *qrng_directions;
    curandDirectionVectors32_t *d_qrng_directions;
    cudaMalloc((void**)&d_qrng_directions,
               3*sizeof(curandDirectionVectors32_t));
    curandGetDirectionVectors32(&qrng_directions,
                                CURAND_DIRECTION_VECTORS_32_JOEKUO6);
    cudaMemcpy(d_qrng_directions, qrng_directions,
               3*sizeof(curandDirectionVectors32_t),
               cudaMemcpyHostToDevice);

    // Allocate space for Q-RNG states.
    cudaMalloc((void**)&d_qrng_states,
               3*N * sizeof(curandStateSobol32_t));

    // Allocate space for normals.
    float* d_out;
    cudaMalloc((void**)&d_out, 3*N * sizeof(float));

    // Allocate space for state-reuse output
    float* h_out_reuse_states = new float[3*N];
    // Allocate space for N-states output.
    float* h_out_N_states = new float[3*N];


    /* Generate only as many states as there are threads. Increase offset to
       compensate when necessary so states can be used to generate multiple
       random normals. */

    // Initialize QRNG.
    init_QRNG_reuse_states<<<grid, block>>>(d_qrng_states, d_qrng_directions,
                                            N_per_thread);
    cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
    cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

    // Generate normals.
    gen_normals_reuse_states<<<grid, block>>>(d_out, d_qrng_states,
                                              N, N_per_thread);
    cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
    cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

    cudaMemcpy(h_out_reuse_states, d_out, 3*N * sizeof(float),
               cudaMemcpyDeviceToHost);


    /* Generate as many states as there are numbers to be generated, without
     * looping.
     */

    // Set such that there are at least as many unique thread IDs as work to
    // be done
    grid.x = (N + block.x - 1) / block.x;

    // Initialize QRNG.
    init_QRNG_no_loop<<<grid, block>>>(d_qrng_states, d_qrng_directions, N);
    cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
    cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

    // Generate normals.
    gen_normals_no_loop<<<grid, block>>>(d_out, d_qrng_states, N);
    cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
    cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

    cudaMemcpy(h_out_N_states, d_out, 3*N * sizeof(float),
               cudaMemcpyDeviceToHost);


    /* Loop through both method's output and check they are identical. */

    bool success = true;
    for (size_t i=0; i<N; i++) {
        if (h_out_reuse_states[i+0*N] != h_out_N_states[i+0*N]) {
            std::cout << "i = " << i << " x values do not agree." << std::endl;
            success = false;
        }
        if (h_out_reuse_states[i+1*N] != h_out_N_states[i+1*N]) {
            std::cout << "i = " << i << " y values do not agree." << std::endl;
            success = false;
        }
        if (h_out_reuse_states[i+2*N] != h_out_N_states[i+2*N]) {
            std::cout << "i = " << i << " z values do not agree." << std::endl;
            success = false;
        }
        std::cout << h_out_N_states[i] << ", " << h_out_N_states[i+N] << ", "
                  << h_out_N_states[i+2*N] << std::endl;
    }

    if (success) {
        std::cout << "All random normals match!" << std::endl;
    }

    cudaFree(d_qrng_states);
    cudaFree(d_qrng_directions);
    cudaFree(d_out);
    delete [] h_out_reuse_states;
    delete [] h_out_N_states;
}
