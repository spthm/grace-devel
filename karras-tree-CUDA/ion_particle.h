#pragma once

namespace grace {

struct ion_particle
{
    // unsigned int id;
    // If we need ID, can instead store mass_H and mass_He. Then this struct
    // still divides evenly into (float)4s.
    float mass, frac_H, frac_He;
    float x_HI, x_HeI, x_HeII;
    float gamma_HI, gamma_HeI, gamma_HeII;
    float H_HI, H_HeI, H_HeII;
};

} // namespace grace
