#pragma once

#include "grace/config.h"

#include <assert.h>
#include <iostream>

#ifdef GRACE_DEBUG
// assert(a > b && "Helpful message") generates warnings from nvcc. The below
// is a more portable, slightly less useful alternative.
#define GRACE_ASSERT_NOMSG(predicate) { assert(predicate); }
#define GRACE_ASSERT_MSG(predicate, err) { const bool err = true; assert(err && (predicate)); }
#define GRACE_SELECT_ASSERT(arg1, arg2, ASSERT_MACRO, ...) ASSERT_MACRO

// Usage:
// GRACE_ASSERT(a > b);
// GRACE_ASSERT(a > b, some_unused_variable_name_as_error_message);
// Where the error 'message' must be a valid, unused variable name.
#define GRACE_MSVC_VAARGS_FIX( x ) x
#define GRACE_ASSERT(...) GRACE_MSVC_VAARGS_FIX(GRACE_SELECT_ASSERT(__VA_ARGS__, GRACE_ASSERT_MSG, GRACE_ASSERT_NOMSG)(__VA_ARGS__))
#else
#define GRACE_ASSERT(...)
#endif // GRACE_DEBUG

template <bool> struct grace_static_assert;
template <> struct grace_static_assert<true> {};
#if __cplusplus >= 201103L || defined(static_assert) || __STDC_VERSION >= 201112L
#define GRACE_STATIC_ASSERT(predicate, msg) { static_assert(predicate, msg); }
#else
#define GRACE_STATIC_ASSERT(predicate, ignore) { grace_static_assert<predicate>(); }
#endif

#define GRACE_GOT_TO() std::cerr << "At " << __FILE__ << "@" << __LINE__ << std::endl;
