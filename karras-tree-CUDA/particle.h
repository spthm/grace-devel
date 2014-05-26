#pragma once

namespace grace {

// Particle position + hsml is represented as a float4 (or double4).
// typedef float4 particle;

// Particle specific properties. Similar to RT_Cell_struct, but compacted and
// without rates.
struct particle_ion
{
    float n_H, n_He, rho;
    float T, entropy;
    float x_HI, x_HeI, x_HeII;
    // float tau_HI; // For caseA/B switching.
};

// Similar to RT_Rates_struct.  Can apply to a single particle during an update,
// or to a ray-particle intersection.  Multiple ray-particle intersections
// may map to the same particle (i.e. a particle can be hit by multiple rays).
struct particle_rates
{
    float gamma_HI, gamma_HeI, gamma_HeII;
    float H_HI, H_HeI, H_HeII;
    float L, L_H1, L_He1, L_He2, L_eH, L_C;
    float alpha_H1, alpha_He1, alpha_He2;
};

// Properties calculated for each ray-particle intersection.  A hybrid of
// RT_Rates_struct and RT_Cell_struct.
// Multiple intersect structs may map to the same particle(_ion) (i.e. the
// same particle may be intersected by multiple rays).
//
// If this is scan-summed to convert from intersection/particle NCols to
// cumulative Ncols, the operator should preserve the rates of the RHS.
// The operator is then associative, but not commutative, so still fits the
// model prescribed by Thrust for custom functors provided to parallel scans.
//
// For now, simple arrays for cum_NCol_HI/HeI/HeII will be used.
// struct intersect
// {
//     particle_rates rates;

//     float column_HI, column_HeI, column_HeII;

// };

} // namespace grace
