#include <GL/glxew.h>

#include <stdint.h>


#include <FL/Fl.H>
#include <FL/gl.h>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Menu_Bar.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_File_Chooser.H>

#include <s3d++/util/fps.h>
#include <s3d++/util/file.h>
#include <s3d++/util/pointer.h>
#include <s3d++/math/r2/param_coord.h>
#include <s3d++/math/r3/proj_transform.h>
#include <s3d++/math/r3/linear_transform.h>
#include <s3d++/math/r3/unit_vector.h>
#include <s3d++/math/r3/point.h>
#include <s3d++/math/r3/box.h>
#include <s3d++/math/r3/matrix.h>
#include <s3d++/math/r3/frustum.h>
#include <s3d++/math/trackball.h>
#include <s3d++/math/rotation.h>
#include <s3d++/util/time.h>
#include <s3d++/util/scoped_value.h>
#include <s3d++/color/names.h>
#include <s3d++/color/hsv.h>
#include <s3d++/color/rgb.h>
#include <s3d++/color/util.h>
#include <s3d++/image/io.h>
#include <s3d++/image/image.h>
#include <s3d++/mesh/face.h>
#include <s3d++/mesh/face_view.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#ifdef True
#	undef True
#endif
#ifdef False
#	undef False
#endif
#ifdef None
#	undef None 
#endif
#ifdef Status
#	undef Status
#endif
#include <OpenMesh/Core/Mesh/TriMeshT.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <boost/optional.hpp>

#include <rply/rply.h>

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>
