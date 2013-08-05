#ifndef S3D_MESH_FACE_VIEW_H
#define S3D_MESH_FACE_VIEW_H

#include "face.h"
#include "../util/pointer.h"

namespace s3d { namespace math
{

namespace detail/*{{{*/
{
    template <class T>
        struct get_base_type
        {
            typedef T type;
        };

    template <class T>
        struct get_base_type<T&> : get_base_type<T>
        {
        };

    template <class T>
        struct get_base_type<std::reference_wrapper<T>> : get_base_type<T>
        {
        };

    template <class T>
        struct get_base_type<const std::reference_wrapper<T>> : get_base_type<T>
        {
        };

    template <class T>
        struct get_base_type<std::shared_ptr<T>> : get_base_type<T>
        {
        };

    template <class T>
        struct get_base_type<const std::shared_ptr<T>> : get_base_type<T>
        {
        };

    template <class T>
        struct get_base_type<std::unique_ptr<T>> : get_base_type<T>
        {
        };

    template <class T>
        struct get_base_type<const std::unique_ptr<T>> : get_base_type<T>
        {
        };

    template <class T>
        auto get_pointer_param(T &&p) 
        -> typename std::enable_if<is_pointer_like<T>::value ||
        is_reference_wrapper<T>::value,
        std::shared_ptr<typename value_type<T>::type>>::type
        {
            return to_pointer(std::forward<T>(p));
        }

    template <class T>
        auto get_pointer_param(T &&p) 
        -> typename std::enable_if<!is_pointer_like<T>::value &&
        !is_reference_wrapper<T>::value,
        std::shared_ptr<typename std::remove_reference<T>::type>>::type
        {
            return std::make_shared<typename std::remove_reference<T>::type>(std::forward<T>(p));
        }

    template <class T>
        auto get_pointer_ref_param(T &&p) 
        -> typename std::enable_if<is_pointer_like<T>::value ||
        is_reference_wrapper<T>::value,
        std::shared_ptr<typename value_type<T>::type>>::type
        {
            return get_pointer_param(std::forward<T>(p));
        }

    template <class T>
        auto get_pointer_ref_param(T &&p) 
        -> typename std::enable_if<!is_pointer_like<T>::value &&
        !is_reference_wrapper<T>::value,
        std::shared_ptr<typename std::remove_reference<T>::type>>::type
        {
            return std::shared_ptr<typename std::remove_reference<T>::type>(&p, null_deleter);
        }
}/*}}}*/

template <class T, int D>
class face_view
{
    template <class IT>
        struct vertex_iterator_adaptor
        : public boost::iterator_adaptor<
          vertex_iterator_adaptor<IT>,
          IT,
          typename copy_const<decltype(*std::declval<IT>()),T>::type>
    {
        typedef typename copy_const<decltype(*std::declval<IT>()), 
        face_view>::type face_view_type;

        typedef typename vertex_iterator_adaptor::iterator_adaptor_ 
            adaptor;
    public:

        vertex_iterator_adaptor() 
            : adaptor(0)
              , m_view(NULL) {}

        vertex_iterator_adaptor(IT it, face_view_type &view)
            : adaptor(it)
              , m_view(&view) {}

        template <class U, class =
            typename std::enable_if<std::is_convertible<U,IT>::value>::type>
            vertex_iterator_adaptor(const vertex_iterator_adaptor<U> &that)
            : adaptor(that.base())
              , m_view(that.m_view) {}
    private:
        friend class boost::iterator_core_access;
        face_view_type *m_view;

        typename adaptor::reference dereference() const
        {
            assert(m_view != NULL);
            assert((size_t)*this->base() < m_view->coords().size());
            return m_view->coords()[*this->base()];
        }
    };

    typedef typename copy_const
        <
        T,
        std::vector<typename std::remove_const<T>::type>
            >::type container_type;

public:
    typedef typename copy_const<T,face<int,D>>::type face_type;
    typedef typename copy_const<face_type, int>::type index_type;
    typedef T vertex_type;
    typedef T value_type;

    typedef vertex_iterator_adaptor<typename face_type::iterator> 
        iterator;
    typedef vertex_iterator_adaptor<typename face_type::const_iterator> 
        const_iterator;

    typedef vertex_iterator_adaptor<typename face_type::reverse_iterator> 
        reverse_iterator;
    typedef vertex_iterator_adaptor<typename face_type::const_reverse_iterator> 
        const_reverse_iterator;

    template <class F2, class C>
        face_view(F2 &&face, C &&coords)
        : m_coords(detail::get_pointer_ref_param(std::forward<C>(coords)))
          , m_face(detail::get_pointer_param(std::forward<F2>(face)))
    {
        m_own_face = !is_reference_wrapper<F2>::value 
            && !is_pointer_like<F2>::value;
    }

    face_view(const face_view &that)
        : m_coords(that.m_coords)
          , m_own_face(that.m_own_face)
    {
        if(m_own_face)
            m_face.reset(new face_type(that.idxface()));
        else
            m_face = that.m_face;
    }

    template <class U>
        face_view(const face_view<U,D> &that)
        : m_coords(that.m_coords)
          , m_own_face(that.m_own_face)
    {
        if(m_own_face)
            m_face(new face_type(that.idxface()));
        else
            m_face = that.m_face;
    }

    template <class U>
        face_view(face_view<U,D> &&that)
        : m_coords(std::move(that.m_coords))
          , m_face(std::move(that.m_face))
          , m_own_face(that.m_own_face)
    {
    }

    face_view &operator=(const face_view &that)
    {
        m_coords = that.m_coords;
        m_own_face = that.m_own_face;
        if(m_own_face)
            m_face(new face_type(that.idxface()));
        else
            m_face = that.m_face;
    }

    template <class U>
        face_view &operator=(const face_view<U,D> &that)
        {
            m_coords = that.m_coords;
            m_own_face = that.m_own_face;
            if(m_own_face)
                m_face(new face_type(that.idxface()));
            else
                m_face = that.m_face;
        }

    template <class U>
        face_view &operator=(face_view<U,D> &&that)
        {
            m_coords = std::move(that.m_coords);
            m_own_face = that.m_own_face;
            m_face = std::move(that.m_face);
            return *this;
        }

    vertex_type &operator[](size_t i)
    {
        assert(i < idxface().size());
        return *std::next(begin(), i);
    }
    const vertex_type &operator[](size_t i) const
    {
        assert(i < idxface().size());
        return *std::next(begin(), i);
    }

    auto index_at(size_t i) const -> const index_type &
    {
        assert(i < idxface().size());
        return idxface()[i];
    }

    auto index_at(size_t i) -> index_type &
    {
        assert(i < idxface().size());
        return idxface()[i];
    }

    size_t size() const { return idxface().size(); }
    bool empty() const { return idxface().empty() || coords().empty(); }

    template <class X=int, class
        = typename std::enable_if<sizeof(X) && 
        !std::is_const<face_type>::value>::type>
        void clear() { idxface().clear(); }

    vertex_type &front() { return *begin(); }
    const vertex_type &front() const { return *begin(); }

    vertex_type &back() { return *rbegin(); }
    const vertex_type &back() const { return *rbegin(); }

    template <class X=int, class
        = typename std::enable_if<sizeof(X) && 
        !std::is_const<face_type>::value>::type>
        void push_back(const vertex_type &val)
        {
            coords().push_back(val);
            push_back_index(coords().size()-1);
        }

    template <class X=int, class
        = typename std::enable_if<sizeof(X) && 
        !std::is_const<face_type>::value>::type>
        void push_back_index(size_t idx)
        {
            assert(idx < coords().size());
            idxface().push_back(idx);
        }

    template <class X=int, class
        = typename std::enable_if<sizeof(X) && 
        !std::is_const<face_type>::value>::type>
        void erase(iterator it)
        { idxface().erase(it.base()); }

    iterator begin() 
    { return iterator(idxface().begin(), *this); }
    iterator end() 
    { return iterator(idxface().end(), *this); }

    const_iterator begin() const 
    { return const_iterator(idxface().begin(), *this); }
    const_iterator end() const 
    { return const_iterator(idxface().end(), *this); }

    reverse_iterator rbegin() 
    { return reverse_iterator(idxface().rbegin(), *this); }
    reverse_iterator rend() 
    { return reverse_iterator(idxface().rend(), *this); }

    const_reverse_iterator rbegin() const 
    { return const_reverse_iterator(idxface().rbegin(), *this); }
    const_reverse_iterator rend() const 
    { return const_reverse_iterator(idxface().rend(), *this); }

    const face_type &idxface() const { assert(m_face); return *m_face; }
    const container_type &coords() const { assert(m_coords); return *m_coords; }

    face_type &idxface() { assert(m_face); return *m_face; }
    container_type &coords() { assert(m_coords); return *m_coords; }

private:
    std::shared_ptr<container_type> m_coords;
    std::shared_ptr<face_type> m_face;

    bool m_own_face;
};

template <class F, class C>
auto make_face_view(F &&face, C &&coords)
    -> face_view<
        typename copy_const<typename detail::get_base_type<F>::type,
        typename detail::get_base_type<C>::type::value_type>::type,
        detail::get_base_type<F>::type::dim>
 {
     return {std::forward<F>(face), std::forward<C>(coords)};
 }

template <class F, class C>
auto make_const_face_view(F &&face, C &&coords)
    -> face_view<const typename detail::get_base_type<C>::type::value_type,
                 detail::get_base_type<F>::type::dim>
{
    return {std::forward<F>(face), std::forward<C>(coords)};
}

template <class T, int D>
struct is_face<face_view<T,D>>
{
    static const bool value = true;
};

}} // namespace s3d::math

#endif
