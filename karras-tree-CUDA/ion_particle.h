#pragma once

namespace grace {

struct ion_particle
{
    float n;
    float x_HI, x_HeI, x_HeII;
    float gamma_HI, gamma_HeI, gamma_HeII;
    float H_HI, H_HeI, H_HeII;
};

} // namespace grace
