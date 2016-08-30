#include "grace/aabb.h"
#include "grace/vector.h"

#include "helper-unit/assert_macros.h"

#include <iostream>

// Convenient, and having commas in macro arguments requires extra parentheses.
typedef grace::Vector<3, float>  Vector3f;
typedef grace::Vector<3, double> Vector3d;

int main(void)
{
    // These also test the templated conversion constructors.
    grace::AABB<float> aabb(Vector3d(-2, -2, -2), Vector3d(2, 2, 2));
    grace::AABB<float> aabb_s(aabb);
    grace::AABB<float> aabb_svec(aabb);
    grace::AABB<double> aabb_t(aabb);

    ASSERT_EQUAL(aabb.area(), 96.f);
    ASSERT_EQUAL(aabb.center(), Vector3f(0.f, 0.f, 0.f));
    ASSERT_EQUAL(aabb.size(),   Vector3f(4.f, 4.f, 4.f));

    aabb_s.scale(2);
    ASSERT_EQUAL(aabb_s.center(), aabb.center());
    ASSERT_EQUAL(aabb_s.size(), Vector3f(8.f, 8.f, 8.f));

    aabb_svec.scale(Vector3f(2.f, 4.f, 8.f));
    ASSERT_EQUAL(aabb_svec.center(), aabb.center());
    ASSERT_EQUAL(aabb_svec.size(), Vector3f(8.f, 16.f, 32.f));

    aabb_t.translate(Vector3d(8.0, -4.0, 8.0));
    ASSERT_EQUAL(aabb_t.center(), Vector3d(8.0, -4.0, 8.0));
    ASSERT_EQUAL(aabb_t.size(), Vector3d(aabb.size()));

    std::cout << "PASSED" << std::endl;

    return 0;
}
