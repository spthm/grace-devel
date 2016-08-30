#include "grace/vector.h"

#include "helper-unit/assert_macros.h"

#include <iostream>

// Convenient, and having commas in macro arguments requires extra parentheses.
typedef grace::Vector<4, float>  Vector4f;

int main(void)
{
    Vector4f a(2.f, -4.f, 8.f, 16.f);
    Vector4f b(8.0,  4.0, 4.0, 2.f);
    Vector4f c(3.f, 4.f, 12.f, 84.f);

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

    ASSERT_EQUAL(-a, Vector4f(-2.f, 4.f, -8.f, -16.f));
    ASSERT_EQUAL(max_element(a), 16.f);
    ASSERT_EQUAL(min_element(a), -4.f);
    ASSERT_EQUAL(norm2(c), 7225.f);
    ASSERT_EQUAL(norm(c), 85.f);
    ASSERT_EQUAL(normalize(c), (float)(1.0 / 85.0) * c);


    //
    // Vector-vector operations
    //

    ASSERT_EQUAL(dot(a, b), 64.f);
    ASSERT_EQUAL(min(a, b), Vector4f(2.f, -4.f, 4.f, 2.f));
    ASSERT_EQUAL(max(a, b), Vector4f(8.f,  4.f, 8.f, 16.f));


    //
    // Vector-vector arithmetic
    //

    ASSERT_EQUAL(a + b, Vector4f(10.f, 0.f, 12.f, 18.f));
    ASSERT_EQUAL(a - b, Vector4f(-6.f, -8.f, 4.f, 14.f));
    ASSERT_EQUAL(a * b, Vector4f(16.f, -16.f, 32.f, 32.f));
    ASSERT_EQUAL(a / b, Vector4f(0.25f, -1.f, 2.f, 8.f));


    //
    // Vector-scalar arithmetic
    //

    ASSERT_EQUAL(a + 2.f, Vector4f(4.f, -2.f, 10.f, 18.f));
    ASSERT_EQUAL(a - 2.f, Vector4f(0.f, -6.f, 6.f, 14.f));
    ASSERT_EQUAL(a * 2.f, Vector4f(4.f, -8.f, 16.f, 32.f));
    ASSERT_EQUAL(a / 2.f, Vector4f(1.f, -2.f, 4.f, 8.f));


    //
    // Scalar-vector arithmetic
    //

    ASSERT_EQUAL(2.f + a, Vector4f(4.f, -2.f, 10.f, 18.f));
    ASSERT_EQUAL(2.f - a, Vector4f(0.f, 6.f, -6.f, -14.f));
    ASSERT_EQUAL(2.f * a, Vector4f(4.f, -8.f, 16.f, 32.f));
    ASSERT_EQUAL(2.f / a, Vector4f(1.f, -0.5f, 0.25f, 0.125f));

    std::cout << "PASSED" << std::endl;

    return 0;
}
