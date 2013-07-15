#include <stdio.h>


// This is from http://stackoverflow.com/a/14038590/927046
#define CUDA_HANDLE_ERR(code) { cudaErrorCheck((code), __FILE__, __LINE__); }

inline void cudaErrorCheck(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDA Error!\nCode: %s\nFile: %s @ line %d\n", cudaGetErrorString(code), file, line);

    if (abort)
      exit(code);
  }
}
