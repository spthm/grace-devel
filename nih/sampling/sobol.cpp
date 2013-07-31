/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <nih/sampling/sobol.h>
#include <stdio.h>

namespace nih {

void Sobol_sampler::generator_matrices(
    const char* direction_numbers,
    const char* matrix_file)
{
    fprintf(stderr, "processing sobol direction numbers %s... started\n", direction_numbers);

    FILE* in_file = fopen(direction_numbers, "r");
    FILE* out_file = fopen(matrix_file, "w");

    unsigned int m[32];
    unsigned int matrix[32];

    fprintf( out_file, "static unsigned int s_sobolMat[] = {\n");

    // skip the header
    char buf[256];
    fscanf( in_file, "%s %s %s %s", buf, buf + 8, buf + 16, buf + 24 );

    while (!feof( in_file ))
    {
        unsigned int d, a, s;

        if (fscanf( in_file, "%u %u %u", &d, &s, &a ) == 0)
            break;

        fprintf(stderr, "\r  dimension %u, s: %u, a: %u", d, s, a);

        for (unsigned int i = 0; i < s; ++i)
            fscanf( in_file, "%u", &m[i] );

        generator_matrix( a, s, m, matrix, 32u );

        for (unsigned int i = 0; i < 32; ++i)
            fprintf( out_file, "0x%x, ", matrix[i] );
        fprintf( out_file, "\n" );
    }

    fprintf( out_file, "};\n");

    fclose( in_file );
    fclose( out_file );

    fprintf(stderr, "\nprocessing sobol direction numbers %s... done\n", direction_numbers);
}

} // namespace nih
