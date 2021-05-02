struct kernel
{
    operator()(grace::Ray<T> ray)
    {
        // Since we only allow a single ray as input to the kernel and traversal
        // routines, frustumizing has to occur earlier than here...
        // auto frustum = frustumize(ray);
        // Note also that frustums require threads to process consecutive rays,
        // which is not otherwise the case on the GPU.

        // However, frustums will likely never be useful all the way down to
        // leaf nodes and primitives, and so we can constrain the user to
        // dealing with only rays and, possibly, ray packets for their
        // intersection and intersection-processing workloads.

        // What does CR (closest-hit result) look like if we have a ray packet?
        // It needs to have multiple components. Should probably be templated
        // over the T above (grace::Ray<T>). Perhaps better to use a typedef
        // e.g. grace::Ray<T>::float_type, which would likely be equal to T
        // but might resolve to float, double, simd<float, ABI>, or
        // simd<double, ABI>.
        // Alternatives include
        //   grace::Ray<T>::numeric_type (a bit lengthy?)
        //   grace::Ray<T>::value_type (perhaps confusing, since Ray != Vector)
        //   grace::Ray<T>::scalar_type (used by Visionray; IMO initially
        //                               confusing)
        //   grace::Ray<T>::quantity_type (a bit lengthy?)
        //   grace::Ray<T>::arithmetic_type (incompatible with
        //                                   std::is_arithmetic, also a bit
        //                                   lengthy?)
        // but use of float_type makes it clear that floating-point values are
        // expected. I do not foresee support for integer-valued ray directions,
        // origins and lengths ever appearing in GRACE.
        CR cr_init(-1);
        CR cr = grace::closest_intersection(ray, traversal_ctx, cr_init);
        // Equivalent to:
        // CR cr = grace::intersect(grace::closest_intersection_tag, ray,
        //                          traversal_ctx, cr_init);
        // or(?):
        // CR cr = grace::intersect<grace::closest_intersection_tag>(
        //             ray, traversal_ctx, cr_init);

        // What does AR (any-hit result) look like if we have a ray packet?
        // It needs to have multiple components. See discussion above.
        AR ar_init(-1);
        AR ar = grace::any_intersection(ray, traversal_ctx, ar_init);
        // Equivalent to:
        // AR ar = grace::intersect(grace::any_intersection_tag, ray,
        //                          traversal_ctx, ar_init);
        // or(?):
        // AR ar = grace::intersect<grace::any_intersection_tag>(
        //             ray, traversal_ctx, ar_init);


        auto all = grace::all_intersections(ray, traversal_ctx);
        for (IR ir = begin(all); ir != end(all); ++ir)
        {
            // Do work.
        }
    }
};

__global__ void trace()
{
    for (auto primitives : grace::intersections_iterator(ray, traversal_ctx))
    {
        for (auto primitive : primitives)
        {
            // Do work.
        }
    }
}

/* On SIMD rays
 * ------------
 *
 *  - We consider SIMD types to be effectively a fundamental numeric type in
 *    GRACE, whose exact implementation depends on the current compilation
 *    target.
 *  - User intersection functions must be compatible with both Ray<float>
 *    and Ray<vector_float> types, i.e. be SIMD compatible.
 *  - Provide an unpacking iterator for edge-cases where this is actually not
 *    possible. This is fairly trivial in terms of the basic vector_type -> type
 *    unpackers we'd expect to be available from a SIMD library.
 *  - On the CPU, define the (default) SIMD width as the maximum available for
 *    the current architecture.
 *  - On the GPU, define the SIMD width as one always.
 *  - 'SIMDization', e.g. via some simdify(...) function, is hidden-away in
 *    the traversal routine(s); we only deal with individual rays as input.
 *
 * We can use an extant SIMD library, rather than reinventing the wheel. For
 * future-proofing, it may be best to typedef this library's SIMD vector types
 * in the grace:: namespace, i.e.,
 *   namespace grace {
 *     typedef simdlib::float_v float_v;
 *     typedef simdlib::int_v int_v;
 *     // etc.
 *   }
 * for all relevant types. That way we can easily change the underlying SIMD
 * library, or provide our own implementation, without breaking user code.
 * Note that a typedef is just a human-friendly alias, and does not declare any
 * new type. Thus the compiler will just see simdlib::<type> and ADL will still
 * work as expected. That is, we do not need to somehow import all of simdlib's
 * operators into the grace:: namespace. But we should document everything we
 * expect to work for grace::float_v, grace::int_v etc. Perhaps some subset of
 * that which is commonly available from SIMD libraries would suffice.
 *
 * As a potential caveat (which I haven't really looked into), note that ADL
 * doesn't work on function templates if _explicit_ template arguments are used,
 * i.e. (note the lack of a namespace prefix for the function)
 *   some_function_in_simdlib<arg>(some_simdlib_variable);
 * won't work. But I don't think such a use-case is likely. Any explicit
 * template arguments will be applied to the types themselves (e.g. specifying a
 * specific size), not to simdlib's function templates...
 *
 * In any case, anything that won't work via ADL should be recreated in the
 * grace:: namespace. Metaprogramming stuff seems a likely candidate, e.g.
 *   simdlib::width<simdlib::float_v>::value
 * would need to be recreated so that the equivalent grace::width was valid.
 * On the other hand, use of the handy new contexpr functions, presumably, means
 * that ADL still works, i.e. (again, note the lack of a namespace prefix for
 * the function)
 *   width(some_simdlib_variable);
 * will work just fine, and should be documented but not recreated.
 *
 * On the current CUDA packets
 * ---------------------------
 *
 * They are fundamentally an architecture-specific optimization and have no real
 * analogue in CPU code, despite (likely) being effective where SIMD packets are
 * effective on the CPU. It would therefore be preferable not to conflate them
 * with SIMD packets.
 * Further, at present, we are considering the low-level API which developers
 * call from their own host functions or CUDA kernels. It then seems perfectly
 * reasonable to allow some architecture-specific divergences in the API that
 * relate to optimization!
 * To that end, have the low-level CUDA interface accept an extra argument based
 * on the new CUDA 9 co-operative groups. A single-thread group (or, for
 * convenience, no additional argument) implies individual-ray traversal, while
 * a warp-size (or, actually, any size greater than one up to and including the
 * warp size) group implies 'packet' traversal as defined on GPUs.
 * This fits nicely with the new
 *   cg::thread_block_tile<32> warp
 *     = tiled_partition<32>(this_thread_block());
 * Using the compile-time tiles exposes additional (and, here, required)
 * functionality such as any(), ballot() and shfl().
 * Note also the alternative
 *   cg::coalesed_group g = cooperative_groups::coalesced_threads();
 * which might also be useful, e.g. if a developer using GRACE in their own CUDA
 * kernels doesn't, or cannot easily, know the current state of the warp at the
 * point GRACE is called. This is in fact often the case in ray tracing, where
 * we might want to trace some shadow rays for only the subset of rays in the
 * warp which hit something, etc.
 *
 * It also appears that, on Volta, threads within a warp can follow different
 * execution paths _simultaneously_, and hence in the above example there is
 * something to be gained when dynamically determining
 * the set of currently-active threads, rather than forcing the entire
 * warp to take part and masking-off those threads which are actually
 * inactive (i.e. do not want to trace shadow rays). However, the current
 * CUDA 9 RC documentation does not make clear if the result of
 * coalesced_threads() has the ballot() and any() intrinsics which are
 * here necessary. (But it certainly does have shfl(), and therefore
 * most likely does have any() and ballot().)
 *
 * Finally, note that if requiring CUDA 9 is too stringent (which will
 * be the case for Eric, since Fermi support has been dropped!) the
 * above thread-management features can all be recreated with extant
 * warp-intrinsics. Thus we should define grace::thread_block_tile<N>
 * and grace::coalesced_group types which inherit from, and can be
 * constructed from (or are merely typedefs for?) their CUDA 9-provided
 * equivalents, but which fall-back to a GRACE-specific definition for
 * CUDA versions < 9. These types then, vaguely, model (a part of) the executors
 * concept that may eventually end up in C++. Specifically, they model a form
 * of stateful execution policy.
 *
 *
 * On executors and the executor model
 * -----------------------------------
 *
 * An Executor is a set of rules governing when, where and how to run a function
 * object. They are lightweight and cheap to copy.
 * An Execution Context is a place where function objects are executed. A thread
 * pool is an example of an Execution Context.
 *
 * The Execution Context has an executor; again, for a thread pool, this
 * executor embodies the rule that code is executed only in the pool. It may
 * have additional state, such as timers or queues.
 *
 * For our CUDA coalesced-threads correspond to an Execution Context which also
 * provides synchronization. (The context is the warp, or some subset of it.
 * Each thread in the warp, or subset, is an executor.)
 * Note that, in the standard proposals, an Execution Context is considered a
 * heavyweight and long-lived object, and is therefore noncopyable. This does
 * not correspond well to our CUDA thread group. But that's probably not an
 * issue. We don't necessarily want to be copying the context around anyway,
 * and ultimately the on-device CUDA programming model is somewhat different
 * from the C++, host-side model. We can't expect parallelism concepts to
 * apply equally well to both.
 *
 * Thus, in this model (which will one day, hopefully, appear in C++) the above
 * thread groups are more-or-less a form of executor; they specify the how
 * and, implicitly, where (when is not of concern to us, and asynchronous
 * execution _within_ a CUDA kernel is not a well-defined concept).
 *
 * We can therefore naturally extend things to the host-side with parallel,
 * vectorized, and parallel-vectorized executors! Parallel implies execution
 * across multiple threads, and vectorized imples use of SIMD rays. Note that
 * one does not imply the other, and that both are in principle configurable:
 * the number of threads and the width of the SIMD vector.
 *
 *  - More advanced traversal techniques, e.g. frustum traversal,



