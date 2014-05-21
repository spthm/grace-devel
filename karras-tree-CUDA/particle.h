#pragma once

namespace grace {

// Particle position + hsml is represented as a float4 (or double4).
// typedef float4 particle;

// Particle specific properties. Similar to RT_Cell_struct, but compacted and
// without rates.
struct ion_particle
{
    float n_H, n_He, rho;
    float T, entropy;
    float x_HI, x_HeI, x_HeII;
    // float tau_HI; // For caseA/B switching.
};

// Similar to RT_Rates_struct.  Can apply to a single particle during an update,
// or to a ray-particle intersection.  Multiple ray-particle intersections
// can map to a single particle (i.e. a particle can be hit by multiple rays).
struct particle_rates
{
    float gamma_HI, gamma_HeI, gamma_HeII;
    float H_HI, H_HeI, H_HeII;
    float L, L_H1, L_He1, L_He2, L_eH, L_C;
    float alpha_H1, alpha_He1, alpha_He2;
};

// Properties calculated for each ray-particle intersection.  A hybrid of
// RT_Rates_struct and RT_Cell_struct.
// Multiple intersect structs may map back to the same (ion_)particle (i.e. the
// same particle may be intersected by multiple rays).
struct intersect
{
    particle_rates rates;

    float column_HI, column_HeI, column_HeII;

};

} // namespace grace
