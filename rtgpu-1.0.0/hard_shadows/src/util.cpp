#include "pch.h"
#include "util.h"

void check_glerror()
{
    std::ostringstream errors;
    GLenum err;
    while((err = glGetError()) != GL_NO_ERROR)
        errors << (const char *)gluErrorString(err) << '\n';

    if(!errors.str().empty())
        throw std::runtime_error("OpenGL: "+errors.str());
};

void check_cuda_error(const std::string &msg)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        if(msg.empty())
            throw std::runtime_error(cudaGetErrorString(err));
        else
        {
            std::stringstream ss;
            ss << msg << ": " << cudaGetErrorString(err);
            throw std::runtime_error(ss.str().c_str());
        }
    }
}
