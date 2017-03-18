#include "read_ply.hpp"
// The ply.h header is C++ compatible, but including it in a .cu file appears
// to break nvcc (likely the Int8 ... Float64 macros).
#include "ply.h"

#include <iostream>
#include <stdio.h>
#include <vector>

/////////////////////////////////
// Vertex and face definitions
/////////////////////////////////

// ~Prevent external linkage.
namespace {

struct PLYVertex {
  float x, y, z;
  // Don't care about anything else.
  void *other;
};

struct PLYFace {
  unsigned char nverts;
  int *vertex_indices;
  // Don't care about anything else.
  void *other;
};

typedef std::vector<PLYVertex> VertexListT;
typedef std::vector<PLYFace> FaceListT;

PlyProperty vertex_properties[] = {
  {(char*)"x", Float32, Float32, offsetof(PLYVertex, x), 0, 0, 0, 0},
  {(char*)"y", Float32, Float32, offsetof(PLYVertex, y), 0, 0, 0, 0},
  {(char*)"z", Float32, Float32, offsetof(PLYVertex, z), 0, 0, 0, 0},
};

PlyProperty face_properties[] = {
  {(char*)"vertex_indices", Int32, Int32, offsetof(PLYFace, vertex_indices),
   1, Uint8, Uint8, offsetof(PLYFace, nverts)},
};

} // namespace


/////////////////////////////////
// PLY object
/////////////////////////////////

static PlyFile *in_ply;
static PlyOtherProp* vertex_other;
static PlyOtherProp* face_other;


/////////////////////////////////
// PLY file reading
/////////////////////////////////

static int read_file(
    const std::string& ply_fname,
    VertexListT& vlist,
    FaceListT& flist)
{
    FILE* ply_fp = fopen(ply_fname.c_str(), "r");
    if (ply_fp == NULL) {
        std::cerr << "Error opening file " << ply_fname << std::endl;
        return 1;
    }

    in_ply = read_ply(ply_fp);
    // Loop through all element groups (where a group is e.g. vertices or faces)
    for (int i = 0; i < in_ply->num_elem_types; ++i)
    {
        int elem_count;
        /* prepare to read the i'th list of elements */
        char* elem_name = setup_element_read_ply(in_ply, i, &elem_count);

        if (equal_strings((char*)"vertex", elem_name))
        {
            vlist = VertexListT(elem_count);

            // Set up to store vertex x, y and z.
            setup_property_ply(in_ply, &vertex_properties[0]);
            setup_property_ply(in_ply, &vertex_properties[1]);
            setup_property_ply(in_ply, &vertex_properties[2]);

            // Detect vertex properties present in the file which we did not
            // explicitly define in vertex_properties.
            vertex_other = get_other_properties_ply(in_ply,
                                                    offsetof(PLYVertex, other));

            // Read in the vertex elements. Properties we don't care about get
            // dumped in each PLYVertex's "other".
            for (int j = 0; j < elem_count; ++j) {
                get_element_ply(in_ply, (void*)(&vlist[j]));
            }
        }
        else if(equal_strings((char*)"face", elem_name))
        {
            flist = FaceListT(elem_count);

            // Set up to store vertex indices of each face.
            setup_property_ply(in_ply, &face_properties[0]);
            // Detect face properties present in the file which we did not
            // explicitly define in face_properties.
            face_other = get_other_properties_ply(in_ply,
                                                  offsetof(PLYFace, other));

            // Read in the face elements. Properties we don't care about get
            // dumped in each PLYFace's "other".
            for (int j = 0; j < elem_count; ++j) {
                get_element_ply(in_ply, (void*)(&flist[j]));
            }
        }
        else {
            // Get anything that isn't a vertex or face element.
            // Returns a pointer to in_ply->other_elems.
            get_other_element_ply(in_ply);
        }
    }

    // This doesn't actually do anything! Memory leak if there are any other
    // elements present.
    free_other_elements_ply(in_ply->other_elems);

    close_ply(in_ply);
    free_ply(in_ply);

    return 0;
}

int read_triangles(const std::string& ply_fname,
                   std::vector<PLYTriangle>& tris)
{
    VertexListT vlist;
    FaceListT flist;

    int status = read_file(ply_fname, vlist, flist);
    if (status != 0) {
        return status;
    }

    tris.resize(flist.size());
    for (size_t i = 0; i < flist.size(); ++i)
    {
        PLYTriangle tri;
        PLYFace face = flist[i];
        if (face.nverts != 3) {
            std::cerr << "Face " << i << " has " << face.nverts << " vertices"
                      << std::endl;
            return 1;
        }

        PLYVertex v1 = vlist[face.vertex_indices[0]];
        PLYVertex v2 = vlist[face.vertex_indices[1]];
        PLYVertex v3 = vlist[face.vertex_indices[2]];

        tri.v1 = grace::Vector<3, float>(v1.x, v1.y, v1.z);
        tri.v2 = grace::Vector<3, float>(v2.x, v2.y, v2.z);
        tri.v3 = grace::Vector<3, float>(v3.x, v3.y, v3.z);

        tris[i] = tri;
    }

    return 0;
}
