import numpy as np
from morton_keys import morton_key_3D

class Ray(object):
    """docstring for Ray"""

    ray_classes = {0: 'MMM',
                   1: 'PMM',
                   2: 'MPM',
                   3: 'PPM',
                   4: 'MMP',
                   5: 'PMP',
                   6: 'MPP',
                   7: 'PPP'}

    def __init__(self, dx, dy, dz, ox=0, oy=0, oz=0, length=2):
        super(Ray, self).__init__()
        N = np.sqrt(dx*dx + dy*dy + dz*dz)

        self.dx = dx / N
        self.dy = dy / N
        self.dz = dz / N

        self.ox = ox
        self.oy = oy
        self.oz = oz

        self.dclass = 0
        if dx > 0:
            self.dclass += 1
        if dy > 0:
            self.dclass += 2
        if dz > 0:
            self.dclass += 4

        self.length = length

        self.key = morton_key_3D(dx, dy, dz)

        self.xbyy = dx / dy
        self.ybyx = 1.0 / self.xbyy
        self.ybyz = dy / dz
        self.zbyy = 1.0 / self.ybyz
        self.xbyz = dx / dz
        self.zbyx = 1.0 / self.xbyz

        self.c_xy = oy - self.ybyx*ox
        self.c_xz = oz - self.zbyx*ox
        self.c_yx = ox - self.xbyy*oy
        self.c_yz = oz - self.zbyy*oy
        self.c_zx = ox - self.xbyz*oz
        self.c_zy = oy - self.ybyz*oz


    def MMM_hit(self, box):
        if ((self.ox < box.bottom[0]) or (self.oy < box.bottom[1]) or (self.oz < box.bottom[2])):
            return False # AABB entirely in wrong octant wrt ray origin

        elif (box.top[0] - self.ox - self.dx*self.length < 0 or
              box.top[1] - self.oy - self.dy*self.length < 0 or
              box.top[2] - self.oz - self.dz*self.length < 0):
            return False # past length of ray

        elif ((self.ybyx * box.bottom[0] - box.top[1] + self.c_xy > 0) or
              (self.xbyy * box.bottom[1] - box.top[0] + self.c_yx > 0) or
              (self.ybyz * box.bottom[2] - box.top[1] + self.c_zy > 0) or
              (self.zbyy * box.bottom[1] - box.top[2] + self.c_yz > 0) or
              (self.zbyx * box.bottom[0] - box.top[2] + self.c_xz > 0) or
              (self.xbyz * box.bottom[2] - box.top[0] + self.c_zx > 0)):
            return False
        return True

    def PMM_hit(self, box):
        if ((self.ox > box.top[0]) or (self.oy < box.bottom[1]) or (self.oz < box.bottom[2])):
            return False # AABB entirely in wrong octant wrt ray origin

        elif (box.bottom[0] - self.ox - self.dx*self.length > 0 or
              box.top[1] - self.oy - self.dy*self.length < 0 or
              box.top[2] - self.oz - self.dz*self.length < 0):
            return False # past length of ray

        elif ((self.ybyx * box.top[0] - box.top[1] + self.c_xy > 0) or
              (self.xbyy * box.bottom[1] - box.bottom[0] + self.c_yx < 0) or
              (self.ybyz * box.bottom[2] - box.top[1] + self.c_zy > 0) or
              (self.zbyy * box.bottom[1] - box.top[2] + self.c_yz > 0) or
              (self.zbyx * box.top[0] - box.top[2] + self.c_xz > 0) or
              (self.xbyz * box.bottom[2] - box.bottom[0] + self.c_zx < 0)):
            return False
        return True

    def MPM_hit(self, box):
        if ((self.ox < box.bottom[0]) or (self.oy > box.top[1]) or (self.oz < box.bottom[2])):
            return False # AABB entirely in wrong octant wrt ray origin

        elif (box.top[0] - self.ox - self.dx*self.length < 0 or
              box.bottom[1] - self.oy - self.dy*self.length > 0 or
              box.top[2] - self.oz - self.dz*self.length < 0):
            return False # past length of ray

        elif ((self.ybyx * box.bottom[0] - box.bottom[1] + self.c_xy < 0) or
              (self.xbyy * box.top[1] - box.top[0] + self.c_yx > 0) or
              (self.ybyz * box.bottom[2] - box.bottom[1] + self.c_zy < 0) or
              (self.zbyy * box.top[1] - box.top[2] + self.c_yz > 0) or
              (self.zbyx * box.bottom[0] - box.top[2] + self.c_xz > 0) or
              (self.xbyz * box.bottom[2] - box.top[0] + self.c_zx > 0)):
            return False
        return True

    def PPM_hit(self, box):
        if ((self.ox > box.top[0]) or (self.oy > box.top[1]) or (self.oz < box.bottom[2])):
            return False # AABB entirely in wrong octant wrt ray origin

        elif (box.bottom[0] - self.ox - self.dx*self.length > 0 or
              box.bottom[1] - self.oy - self.dy*self.length > 0 or
              box.top[2] - self.oz - self.dz*self.length < 0):
            return False # past length of ray

        elif ((self.ybyx * box.top[0] - box.bottom[1] + self.c_xy < 0) or
              (self.xbyy * box.top[1] - box.bottom[0] + self.c_yx < 0) or
              (self.ybyz * box.bottom[2] - box.bottom[1] + self.c_zy < 0) or
              (self.zbyy * box.top[1] - box.top[2] + self.c_yz > 0) or
              (self.zbyx * box.top[0] - box.top[2] + self.c_xz > 0) or
              (self.xbyz * box.bottom[2] - box.bottom[0] + self.c_zx < 0)):
            return False
        return True

    def MMP_hit(self, box):
        if ((self.ox < box.bottom[0]) or (self.oy < box.bottom[1]) or (self.oz > box.top[2])):
            return False # AABB entirely in wrong octant wrt ray origin

        elif (box.top[0] - self.ox - self.dx*self.length < 0 or
              box.top[1] - self.oy - self.dy*self.length < 0 or
              box.bottom[2] - self.oz - self.dz*self.length > 0):
            return False # past length of ray

        elif ((self.ybyx * box.bottom[0] - box.top[1] + self.c_xy > 0) or
              (self.xbyy * box.bottom[1] - box.top[0] + self.c_yx > 0) or
              (self.ybyz * box.top[2] - box.top[1] + self.c_zy > 0) or
              (self.zbyy * box.bottom[1] - box.bottom[2] + self.c_yz < 0) or
              (self.zbyx * box.bottom[0] - box.bottom[2] + self.c_xz < 0) or
              (self.xbyz * box.top[2] - box.top[0] + self.c_zx > 0)):
            return False
        return True

    def PMP_hit(self, box):
        if ((self.ox > box.top[0]) or (self.oy < box.bottom[1]) or (self.oz > box.top[2])):
            return False # AABB entirely in wrong octant wrt ray origin

        elif (box.bottom[0] - self.ox - self.dx*self.length > 0 or
              box.top[1] - self.oy - self.dy*self.length < 0 or
              box.bottom[2] - self.oz - self.dz*self.length > 0):
            return False # past length of ray

        elif ((self.ybyx * box.top[0] - box.top[1] + self.c_xy > 0) or
              (self.xbyy * box.bottom[1] - box.bottom[0] + self.c_yx < 0) or
              (self.ybyz * box.top[2] - box.top[1] + self.c_zy > 0) or
              (self.zbyy * box.bottom[1] - box.bottom[2] + self.c_yz < 0) or
              (self.zbyx * box.top[0] - box.bottom[2] + self.c_xz < 0) or
              (self.xbyz * box.top[2] - box.bottom[0] + self.c_zx < 0)):
            return False

        return True

    def MPP_hit(self, box):
        if ((self.ox < box.bottom[0]) or (self.oy > box.top[1]) or (self.oz > box.top[2])):
            return False # AABB entirely in wrong octant wrt ray origin

        elif (box.top[0] - self.ox - self.dx*self.length < 0 or
              box.bottom[1] - self.oy - self.dy*self.length > 0 or
              box.bottom[2] - self.oz - self.dz*self.length > 0):
            return False # past length of ray

        elif ((self.ybyx * box.bottom[0] - box.bottom[1] + self.c_xy < 0) or
              (self.xbyy * box.top[1] - box.top[0] + self.c_yx > 0) or
              (self.ybyz * box.top[2] - box.bottom[1] + self.c_zy < 0) or
              (self.zbyy * box.top[1] - box.bottom[2] + self.c_yz < 0) or
              (self.zbyx * box.bottom[0] - box.bottom[2] + self.c_xz < 0) or
              (self.xbyz * box.top[2] - box.top[0] + self.c_zx > 0)):
            return False

        return True

    def PPP_hit(self, box):
        if ((self.ox > box.top[0]) or (self.oy > box.top[1]) or (self.oz > box.top[2])):
            return False # AABB entirely in wrong octant wrt ray origin

        elif (box.bottom[0] - self.ox - self.dx*self.length > 0 or
              box.bottom[1] - self.oy - self.dy*self.length > 0 or
              box.bottom[2] - self.oz - self.dz*self.length > 0):
            return False # past length of ray

        elif ((self.ybyx * box.top[0] - box.bottom[1] + self.c_xy < 0) or
              (self.xbyy * box.top[1] - box.bottom[0] + self.c_yx < 0) or
              (self.ybyz * box.top[2] - box.bottom[1] + self.c_zy < 0) or
              (self.zbyy * box.top[1] - box.bottom[2] + self.c_yz < 0) or
              (self.zbyx * box.top[0] - box.bottom[2] + self.c_xz < 0) or
              (self.xbyz * box.top[2] - box.bottom[0] + self.c_zx < 0)):
            return False

        return True

    def AABB_hit(self, box):
        dclass_hit_name = self.ray_classes[self.dclass] + '_hit'
        dclass_hit_fn = getattr(self, dclass_hit_name)
        return dclass_hit_fn(box)

    def sphere_hit(self, x, y, z, radius):
        # Ray origin -> sphere centre.
        px = x - self.ox
        py = y - self.oy
        pz = z - self.oz

        # Normalized ray direction.
        rx = self.dx
        ry = self.dy
        rz = self.dz

        # Projection of p onto r.
        dot_p = px*rx + py*ry + pz*rz

        # Impact parameter.
        bx = px - dot_p*rx
        by = py - dot_p*ry
        bz = pz - dot_p*rz
        b = np.sqrt(bx*bx + by*by +bz*bz)

        if (b >= radius):
            return False

        # If dot_p < 0, to hit the ray origin must be inside the sphere.
        # This is not possible if the distance along the ray (backwards from its
        # origin) to the point of closest approach is > the sphere radius.
        if (dot_p < -radius):
            return False

        # The ray terminates before piercing the sphere.
        if (dot_p > self.length + radius):
            return False

        # Otherwise, assume we have a hit.  This counts the following partial
        # intersections as hits:
        #     i) Ray starts (anywhere) inside sphere.
        #    ii) Ray ends (anywhere) inside sphere.
        return True
