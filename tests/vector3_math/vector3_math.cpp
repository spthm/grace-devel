#include "grace/vector.h"

#include "helper-unit/assert_macros.h"

#include <cmath>
#include <iostream>

// Convenient, and having commas in macro arguments requires extra parentheses.
typedef grace::Vector<3, float>  Vector3f;
typedef grace::Vector<3, double> Vector3d;

const double PI = 3.14159265358979323846;

int main(void)
{
    Vector3f a(2.f, -4.f, 8.f);
    Vector3f b(8.0,  4.0, 4.0);
    Vector3f c(3.f, 4.f, 0.f);
    Vector3f d(1.0, 0.0, 0.0);
    Vector3f e(std::sqrt(2), std::sqrt(2), 0.0);

    /*
     * Note that grace:: is not prefixed to any of the below operators.
     * See argument-dependent name lookup
     */


    //
    // Comparison operators
    //

    ASSERT_EQUAL(a, a);
    ASSERT_NOT_EQUAL(a, b);


    //
    // Vector operations
    //

    ASSERT_EQUAL(-a, Vector3f(-2.f, 4.f, -8.f));
    ASSERT_EQUAL(max_element(a), 8.f);
    ASSERT_EQUAL(min_element(a), -4.f);
    ASSERT_EQUAL(norm2(a), 84.f);
    ASSERT_EQUAL(norm(a), std::sqrt(84.f));
    ASSERT_EQUAL(normalize(c), 0.2f * c);


    //
    // Vector-vector operations
    //

    ASSERT_EQUAL(cross(a, b), Vector3f(-48.f, 56.f, 40.f));
    ASSERT_EQUAL(angular_separation(d, e), (float)(PI / 4.0));
    ASSERT_EQUAL(great_circle_distance(d, e, 1.f), (float)(PI / 4.0));
    ASSERT_EQUAL(dot(a, b), 32.f);
    ASSERT_EQUAL(min(a, b), Vector3f(2.f, -4.f, 4.f));
    ASSERT_EQUAL(max(a, b), Vector3f(8.f,  4.f, 8.f));


    //
    // Vector-vector arithmetic
    //

    ASSERT_EQUAL(a + b, Vector3f(10.f, 0.f, 12.f));
    ASSERT_EQUAL(a - b, Vector3f(-6.f, -8.f, 4.f));
    ASSERT_EQUAL(a * b, Vector3f(16.f, -16.f, 32.f));
    ASSERT_EQUAL(a / b, Vector3f(0.25f, -1.f, 2.f));


    //
    // Vector-scalar arithmetic
    //

    ASSERT_EQUAL(a + 2.f, Vector3f(4.f, -2.f, 10.f));
    ASSERT_EQUAL(a - 2.f, Vector3f(0.f, -6.f, 6.f));
    ASSERT_EQUAL(a * 2.f, Vector3f(4.f, -8.f, 16.f));
    ASSERT_EQUAL(a / 2.f, Vector3f(1.f, -2.f, 4.f));


    //
    // Scalar-vector arithmetic
    //

    ASSERT_EQUAL(2.f + a, Vector3f(4.f, -2.f, 10.f));
    ASSERT_EQUAL(2.f - a, Vector3f(0.f, 6.f, -6.f));
    ASSERT_EQUAL(2.f * a, Vector3f(4.f, -8.f, 16.f));
    ASSERT_EQUAL(2.f / a, Vector3f(1.f, -0.5f, 0.25f));

    std::cout << "PASSED" << std::endl;

    return 0;
}
