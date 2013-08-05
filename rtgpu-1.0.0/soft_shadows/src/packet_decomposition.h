#ifndef PACKET_DECOMPOSITION_H
#define PACKET_DECOMPOSITION_H

template <class T> class dvector;

__host__ void decompose_into_packets(dvector<unsigned> &packet_indices,
                                     dvector<unsigned> &packet_sizes,
                                     dvector<unsigned> *revmap,
                                     const dvector<unsigned> &comp_base,
                                     const dvector<unsigned> &comp_size,
                                     int max_size);

__host__ void decompose_into_packets(dvector<unsigned> &packet_indices,
                                     dvector<unsigned> &packet_sizes,
                                     const dvector<unsigned> &ray_hashes,
                                     int max_size);

__host__ void init_skeleton(dvector<unsigned> &skel, 
                            dvector<unsigned> &head_flags,
                            int skel_value, size_t size);

#endif
