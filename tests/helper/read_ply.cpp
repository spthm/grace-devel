// The ply.h header is C++ compatible.
#include "ply.h"

#include <iostream>
#include <stdio.h>
#include <vector>

/////////////////////////////////
// Vertex and face definitions
/////////////////////////////////

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


/////////////////////////////////
// PLY object
/////////////////////////////////

static PlyFile *in_ply;
static PlyOtherProp* vertex_other;
static PlyOtherProp* face_other;


/////////////////////////////////
// PLY file reading
/////////////////////////////////

int read_file(const char* const ply_fname,
              VertexListT& vlist,
              FaceListT& flist)
{
    FILE* ply_fp = fopen(ply_fname, "r");
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


/////////////////////////////////
// Main
/////////////////////////////////

int main(int argc, char* argv[])
{
    int status = 0;
    char* ply_fname;
    VertexListT vlist;
    FaceListT flist;

    ply_fname = argv[1];

    std::cout << "Reading file " << ply_fname << std::endl;
    status = read_file(ply_fname, vlist, flist);
    if (status != 0) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

