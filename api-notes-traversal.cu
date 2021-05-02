// How do we intersect a ray and a BVH? We traverse it, and intersect the ray
// with nodes of the BVH. Generically, the traversal is unspecified --- we could
// just intersect every node, which would produce correct results but be very
// inefficient.
//
// So, traversal here is the algorithm by which we search a given object for
// intersections. This is provided by the traversal context, which at present
// is either nothing, in which case we apply intersect() to each element of the
// input in turn, or it is a BvhContext, in which case we employ a stack-based
// traversal of the BVH.
//
// This could be extended to other traversal contexts, e.g. breadth-first BVH
// traversal contexts, kd-tree traversal contexts, BVH-N traversal contexts (for
// BVHs of degree N > 2), or multi-domain traversal contexts in an MPI
// framework. In all cases we always call
//   intersect(ray, object_to_be_intersected, traversal_context),
// where we see that we provide the intersect() function with additional
// information governing traversal, which is an orthogonal concept to the
// intersection itself. We should probably actually provide a
// LinearTraversalContext, which is used as the default.
//
// We need to distinguish between intersect(ray, collection_of_primitives) and
// intersect(ray, primitive)... But let's assume all cases of the latter are
// already covered, or are erroneous: either GRACE provides the intersect()
// overload, or the user does. Then the template for the generic intersect
// routine needs only to detect whether T is an iterable/container type (and
// if it is not, to raise an error, e.g. via static_assert("Not implemented")).
// This means checking whether
//
//   using std::begin; // for ADL.
//   using std::end;   // for ADL.
//   begin(instance_of_T);
//   end(instance_of_T);
//
// is valid code. This can be done with something like
//
//   template <typename T, typename = void>
//   struct is_iterable : std::false_type {};
//   template <typename T>
//   struct is_iterable<T, std::void_t<decltype(begin(std::declval<T>())),
//                                     decltype(end(declval<T>()))>>
//       : std::true_type {};
//
// Note std::void_t is >= C++17, but easily implemented in >= C++11. We need
// to re-implement the C++11 std::begin and std::end functions, since they can't
// be called from device code. They call .begin() and .end() for the variable,
// but are overloaded for array-types.
// See e.g. https://github.com/gcc-mirror/gcc/blob/master/libstdc%2B%2B-v3/include/bits/range_access.h
//
// This allows for recursion, where intersect() may call some other intersect()
// method and return its result. So where do we end? Well, on whichever
// intersect() call actually returns a result, rather than returning the result
// of another intersect(). This implies that recursion automatically ends at
// an intersect(ray, sphere) call, i.e. one defined by GRACE, or an
// intersect(ray, user_defined_primitive_instance), i.e. one defined by the
// user.
//
// Traversal contexts should(?) be tagged as such, to allow for flexible and
// extensible static dispatch, e.g.
//   detail::intersect(..., ctxtness<is_traversal_ctx<T>::value>());
//
// So e.g. how to we 'intersect'
//   i) an array of primitives:
//        call intersect(ray, primitive) for each primitive
//   ii) a bvh:
//         call intersect(ray, primitive) for each primitive in each intersected
//         leaf
//   iii) an array of bvh traversal contexts:
//          call intersect(ray, bvh_ctx) on each bvh_ctx in the array

struct kernel
{
    // Allow most parameters except non-const ref.
    operator()(const grace::Ray<T>& ray)
    {
        // I think this will be >= C++17 only, because grace::intersect
        // will have different types for begin() and end() (we want end() to
        // return some lightweight sentinel object, while begin() returns the
        // heavyweight iterator with a traversal stack)...
        for (auto primitive : grace::intersect(ray, traversal_ctx))
        {
            // Do work.
        }

        // So perhaps it would be best not to provide an iterator, or at least
        // to provide it only where >= C++17 is detected.
        // For all other cases, a basic visitor. (Note that auto as the type for
        // a lambda parameter is not allowed until C++14.)
        grace::intersect(ray, traversal_ctx,
        [ray](Primitive primitive)
        {
            // Do work
        });
    }
};

struct kernel
{
    operator()(const grace::Ray<T>& ray)
    {
        // Type of 'primitives' is grace::LinearTraversalCtx<Primitive>
        for (auto primitives : grace::intersect(ray, traversal_ctx.nodes_ctx()))
        {
            for (auto primitive : grace::intersect(ray, primitives))
            {
                // Do work.
            }
        }

        // .nodes_ctx() returns a grace::BvhNodeTraversalCtx instance, which
        // signals the intersection routine to return only intersected leaves
        // --- or, rather, the primitives within them.
        grace::intersect(ray, traversal_ctx.nodes_ctx(),
        [ray](grace::LinearTraversalCtx<Primitive> primitives)
        {
            grace::intersect(ray, primitives,
            [ray](Primitive primitive)
            {
                // Do work.
            });
        });
    }
};

struct kernel
{
    operator()(const grace::Ray<T>& ray)
    {
        // Passing a function object whose argument is LeafTraversalCtx<P>
        // intercepts the hierarchy of grace::intersect() calls wherever
        // grace::intersect(...) would otherwise call
        // grace::intersect(ray, {LeafTraversalCtx<P>} ctx).
        // We can implement this in grace::intersect by checking whether
        // applying the user-provided 'catcher' to the produced traversal
        // context is well-formed. If it is, we do it. If it isn't, we don't,
        // and instead call grace::intersect(ray, next_ctx, user_catcher).
        // This invocation of grace::intersect performs the same check on
        // whatever traversal context it produces.
        // Note that this is mostly compatible with the first kernel above,
        // where a user provides a custom Primitive intersection routine.
        // However, we need to distinguish between a traversal context which is
        // a single primitive to be tested for intersection, and a traversal
        // context which is an intersected primitive to be processed.
        grace::intersect(ray, traversal_ctx,
        [ray](grace::LeafTraversalCtx<Primitive> primitives)
        {
            grace::intersect(ray, primitives,
            [ray](Primitive primitive)
            {
                // Do work.
            });
        });
    }
};
