#pragma once

namespace grace {

struct particle
{
    float4 xyzr;
    unsigned int id;
    float rho;
    float gamma_HI, gamma_HeI, gamma_HeII;
    float sigma_HI, sigma_HeI, sigma_HeII;
};

} // namespace grace
