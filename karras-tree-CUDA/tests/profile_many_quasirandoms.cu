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

__global__ void init_QRNG_loop_states(curandStateSobol32_t *const qrng_states,
                                      curandDirectionVectors32_t *const qrng_directions,
                                      const unsigned int N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    // Initialise the Q-RNG
    // We may have that max(tid) < N, but still generate N states, with each
    // being used only once.

    while (tid < N) {
        curand_init(qrng_directions[0], tid+2, &qrng_states[tid+0*N]); // x
        curand_init(qrng_directions[1], tid+2, &qrng_states[tid+1*N]); // y
        curand_init(qrng_directions[2], tid+2, &qrng_states[tid+2*N]); // z

        tid += stride;
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

__global__ void gen_normals_loop_states(float *normals,
                                        curandStateSobol32_t *const qrng_states,
                                        const size_t N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;

    float x, y, z;

    while (tid < N)
    {
        curandStateSobol32_t x_state = qrng_states[tid+0*N];
        curandStateSobol32_t y_state = qrng_states[tid+1*N];
        curandStateSobol32_t z_state = qrng_states[tid+2*N];

        x = curand_normal(&x_state);
        y = curand_normal(&y_state);
        z = curand_normal(&z_state);

        normals[tid+0*N] = x;
        normals[tid+1*N] = y;
        normals[tid+2*N] = z;

        tid += stride;
    }
}

__global__ void gen_normals_reuse_states(float *normals,
                                         curandStateSobol32_t *const qrng_states,
                                         const size_t N)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
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

        normals[tid+0*N] = x;
        normals[tid+1*N] = y;
        normals[tid+2*N] = z;

        tid += stride;
    }
}

int main(void) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed;
    float total_time = 0;

    // Loop through various numbers of rays to generate.
    const size_t N_trials = 7;
    unsigned int Ns[N_trials] = {1000, 5000, 10000, 50000,
                                 1000000, 2000000, 4000000};

    for (size_t i=0; i<N_trials; i++)
    {
        /* Data common to all methods. */

        unsigned int N = Ns[i];
        unsigned int N_per_thread;

        dim3 grid, block;

        curandStateSobol32_t *d_qrng_states;

        std::cout << "N = " << N << std::endl;

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

        // Allocate space for normals.
        float* d_out;
        cudaMalloc((void**)&d_out, 3*N * sizeof(float));


        /* Generate only as many states as there are threads. Increase offset to
           compensate when necessary so states can be used to generate multiple
           random normals. */

        // Set up for a limited number of threads.
        block.x = 512;
        grid.x = 7;

        // Calculate excess work needed to be done by each thread (upper bound).
        N_per_thread = (N + grid.x*block.x - 1) / (grid.x * block.x);
        std::cout << "N_per_thread = " << N_per_thread << std::endl;

        // Allocate space for 3 * N_threads Q-RNG states.
        cudaMalloc((void**)&d_qrng_states,
                   3*block.x*grid.x * sizeof(curandStateSobol32_t));

        // Initialize QRNG.
        cudaEventRecord(start);
        init_QRNG_reuse_states<<<grid, block>>>(d_qrng_states, d_qrng_directions,
                                                N_per_thread);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time = elapsed;
        cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
        cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

        // Generate normals.
        cudaEventRecord(start);
        gen_normals_reuse_states<<<grid, block>>>(d_out, d_qrng_states, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time += elapsed;
        cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
        cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

        std::cout << "Total time for reuse states method:    " << total_time
                  << std::endl;
        total_time = 0;

        cudaFree(d_qrng_states);


        /* Generate as many states as there are numbers to be generated,
         * looping to compensate for blockDim.x * gridDim.x < N.
         */

        // Allocate space for 3*N Q-RNG states.
        cudaMalloc((void**)&d_qrng_states,
                   3*N * sizeof(curandStateSobol32_t));

        // Initialize QRNG.
        cudaEventRecord(start);
        init_QRNG_loop_states<<<grid, block>>>(d_qrng_states, d_qrng_directions, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time = elapsed;
        cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
        cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

        // Generate normals.
        cudaEventRecord(start);
        gen_normals_loop_states<<<grid, block>>>(d_out, d_qrng_states, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time += elapsed;
        cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
        cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

        std::cout << "Total time for loop states method:     " << total_time
                  << std::endl;
        total_time = 0;

        cudaFree(d_qrng_states);


        /* Generate as many states as there are numbers to be generated, without
         * looping.
         */

        block.x = 512;
        grid.x = (N + block.x - 1) / block.x;

        // Allocate space for N Q-RNG states.
        cudaMalloc((void**)&d_qrng_states,
                   3*N * sizeof(curandStateSobol32_t));

        // Initialize QRNG.
        cudaEventRecord(start);
        init_QRNG_no_loop<<<grid, block>>>(d_qrng_states, d_qrng_directions, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time = elapsed;
        cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
        cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

        // Generate normals.
        cudaEventRecord(start);
        gen_normals_no_loop<<<grid, block>>>(d_out, d_qrng_states, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time += elapsed;
        cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
        cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

        std::cout << "Total time for method no loops method: " << total_time
                  << std::endl;
        std::cout << std::endl;
        total_time = 0;

        cudaFree(d_qrng_states);


        /* Free all data before next run. */

        cudaFree(d_qrng_directions);
        cudaFree(d_out);
    }
}
