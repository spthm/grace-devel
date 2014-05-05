#include <curand_kernel.h>

#include <fstream>

#define N_DIM_MAX 3

inline void cuda_call(cudaError_t code, char* file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error!\nMsg:  %s\nFile: %s @ line %d\n",
                cudaGetErrorString(code), file, line);

    if (abort)
        exit(code);
    }
}

__global__ void init_QRNG(curandStateSobol32_t *const qrng_states,
                          curandDirectionVectors32_t *const qrng_directions, //const?
                          const unsigned int N_dim)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialise the Q-RNG(s).
    for (unsigned int d=0; d<N_dim; d++) {
        curand_init(qrng_directions[d], tid+2, &qrng_states[N_dim*tid+d]);
    }
}


__global__ void gen_normals(float* out,
                            curandStateSobol32_t *const qrng_states,
                            const size_t N,
                            const unsigned int N_dim)
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

    curandStateSobol32_t local_states[N_DIM_MAX];

    for (unsigned int d=0; d<N_dim; d++) {
        local_states[d] = qrng_states[N_dim*tid+d];
    }

    while (tid < N)
    {
        float normal;
        for (unsigned int d=0; d<N_dim; d++) {
            normal = curand_normal(&local_states[d]);
            out[N_dim*tid+d] = normal;
        }

        tid += blockDim.x * gridDim.x;
    }
}

int main(int argc, char* argv[]) {

    std::ofstream outfile;
    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(9);

    size_t N = 100;
    unsigned int N_dim = 1;

    if (argc > 1) {
        N_dim = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N = (size_t) std::strtol(argv[2], NULL, 10);
    }

    // Allocate space for Q-RNG states and the direction vector(s).
    curandStateSobol32_t *d_qrng_states;
    cudaMalloc((void**)&d_qrng_states, 2*32*N_dim * sizeof(curandStateSobol32_t));

    curandDirectionVectors32_t *d_qrng_directions;
    cudaMalloc((void**)&d_qrng_directions,
               N_dim*sizeof(curandDirectionVectors32_t));

    // Generate Q-RNG 'direction vectors' on host, and copy to device.
    curandDirectionVectors32_t *qrng_directions;
    curandGetDirectionVectors32(&qrng_directions,
                                CURAND_DIRECTION_VECTORS_32_JOEKUO6);
    cudaMemcpy(d_qrng_directions, qrng_directions,
               N_dim*sizeof(curandDirectionVectors32_t),
               cudaMemcpyHostToDevice);

    // Initialize QRNG.
    init_QRNG<<<2, 32>>>(d_qrng_states, d_qrng_directions, N_dim);
    cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
    cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

    // Allocate space for normals and generate them.
    float *d_out;
    cudaMalloc((void**)&d_out, N*N_dim*sizeof(float));
    gen_normals<<<2, 32>>>(d_out, d_qrng_states, N, N_dim);
    cuda_call( cudaPeekAtLastError(), __FILE__, __LINE__ );
    cuda_call( cudaDeviceSynchronize(), __FILE__, __LINE__ );

    // Copy back to device.
    float* h_out = new float[N*N_dim];
    cudaMemcpy(h_out, d_out, N*N_dim*sizeof(float), cudaMemcpyDeviceToHost);

    // Write normals to file.
    outfile.open("normals.txt");
    for (size_t i=0; i<N; i++) {
        for (unsigned int d=0; d<N_dim; d++) {
            outfile << h_out[N_dim*i+d] << " ";
            if (h_out[N_dim*i+d] > 0)
                outfile << " ";
        }
        outfile << std::endl;
    }
    outfile.close();

    cudaFree(d_qrng_states);
    cudaFree(d_qrng_directions);
    cudaFree(d_out);
}
