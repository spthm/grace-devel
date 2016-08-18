#pragma once

#include "grace/sphere.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

typedef grace::Sphere<float> SphereType;

//-----------------------------------------------------------------------------
// Utilities for reading in Gadget-2 (type 1) files
//-----------------------------------------------------------------------------

struct gadget_header
{
  int npart[6];
  double mass[6];
  // double time;
  // double redshift;
  // int flag_sfr;
  // int flag_feedback;
  // int npartTotal[6];
  // int flag_cooling;
  // int num_files;
  // double BoxSize;
  // double Omega0;
  // double OmegaLambda;
  // double HubbleParam;

  // fill to 256 Bytes
  // char fill[256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8];
  char fill [256 - 6*4 - 6*8];
};

inline void skip_n(std::ifstream& file, const int n)
{
    int dummy;
    for (int i = 0; i < n; ++i) {
        file.read((char*)&dummy, 4);
    }
}

inline void skip_blockmarker(std::ifstream& file)
{
    skip_n(file, 1);
}

inline void skip_block(std::ifstream& file, const int* npart, const int Npp)
{
    skip_blockmarker(file);
    for (int ptype = 0; ptype < 6; ++ptype) {
        skip_n(file, npart[ptype] * Npp);
    }
    skip_blockmarker(file);
}

inline gadget_header read_gadget_header(std::ifstream& file)
{
    gadget_header header;
    skip_blockmarker(file);
    file.read((char*)&header.npart, 6 * sizeof(int));
    file.read((char*)&header.mass, 6 * sizeof(double));
    file.read((char*)&header.fill, sizeof(header.fill));
    skip_blockmarker(file);
    return header;
}

inline void read_gadget(const std::string fname,
                        thrust::host_vector<SphereType>& h_spheres)
{
    const int GAS_TYPE = 0;
    int N_withmass = 0;

    std::ifstream file;
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    file.open(fname.c_str(), std::ios::binary);
    const gadget_header header = read_gadget_header(file);

    // Calculate particle number counts and read in positions block for gas
    // particles only.
    int N_gas = header.npart[0];
    h_spheres.resize(N_gas);

    if (N_gas == 0) {
        std::stringstream msg_stream;
        msg_stream << "Gadget file " << fname << " has no gas particles!";
        const std::string msg = msg_stream.str();
        throw std::runtime_error(msg);
    }

    float f_dummy;
    skip_blockmarker(file);
    for(int i = 0; i < 6; ++i)
    {
        if (header.mass[i] == 0) {
            N_withmass += header.npart[i];
        }

        for(int n = 0; n < header.npart[i]; ++n)
        {
            if (i == GAS_TYPE) {
                file.read((char*)&h_spheres[n].x, sizeof(float));
                file.read((char*)&h_spheres[n].y, sizeof(float));
                file.read((char*)&h_spheres[n].z, sizeof(float));
            }
            else {
                file.read((char*)&f_dummy, sizeof(float));
                file.read((char*)&f_dummy, sizeof(float));
                file.read((char*)&f_dummy, sizeof(float));
            }
        }
    }
    skip_blockmarker(file);

    // Velocities.
    skip_block(file, header.npart, 3);

    // IDs.
    skip_block(file, header.npart, 1);

    // Masses (optional).  Spacers only exist if the block exists.
    // Otherwise, all particles of a given type have equal mass, saved in the
    // header.
    if (N_withmass > 0)
    {
        skip_blockmarker(file);
        for(int i = 0; i < 6; ++i)
        {
            if (header.mass[i] == 0) {
                skip_n(file, header.npart[i]);
            }
        }
        skip_blockmarker(file);
    }

    // Gas properties (optional).
    if (N_gas > 0)
    {
        // Internal energies.
        skip_blockmarker(file);
        skip_n(file, N_gas);
        skip_blockmarker(file);

        // Densities.
        skip_blockmarker(file);
        skip_n(file, N_gas);
        skip_blockmarker(file);

        // Smoothing lengths.
        skip_blockmarker(file);
        for (int n = 0; n < N_gas; ++n) {
            file.read((char*)&h_spheres[n].r, sizeof(float));
        }
        skip_blockmarker(file);
    }

    file.close();
}

inline void read_gadget(const std::string fname,
                        thrust::device_vector<SphereType>& d_spheres)
{
    thrust::host_vector<SphereType> h_spheres;
    read_gadget(fname, h_spheres);
    d_spheres = h_spheres;
}
