#ifndef RAY_COMPRESSION_H
#define RAY_COMPRESSION_H

template <class T> class dvector;

__host__ void compress_rays(dvector<unsigned> &chunk_hash,
                            dvector<unsigned> &chunk_base,
                            dvector<unsigned> *chunk_idx,
                            const dvector<unsigned> &ray_hashes);

#endif
