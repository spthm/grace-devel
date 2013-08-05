#ifndef S3D_IMAGE_LARGE_IMAGE_H
#define S3D_IMAGE_LARGE_IMAGE_H

#include <memory>
#include "../math/r2/size.h"
#include "../util/clonable.h"
#include "../util/movable.h"
#include "format/format.h"
#include "params.h"
#include "fwd.h"

namespace s3d { namespace img
{

class large_image_impl
{
public:
	large_image_impl(const r2::usize &s, const image_traits<> &traits) 
		: m_size(s), m_traits(traits) {}
	virtual ~large_image_impl() {}

	std::unique_ptr<Image<>> read(const r2::ibox &bounds) const;

	void write(const r2::ipoint &topleft, const Image<> &img,
			   const r2::ibox &bounds_img)
		{ return do_write(topleft, img, bounds_img); }

	void write(const r2::ibox &bounds, const color::radiance &c)
		{ return do_write(bounds, c); }

	const r2::usize &size() const { return m_size; }
	const size_t &w() const { return size().w; }
	const size_t &h() const { return size().h; }

	const image_traits<> &traits() const { return m_traits; }

private:
	r2::usize m_size;
	const image_traits<> &m_traits;

	virtual std::unique_ptr<Image<>> do_read(const r2::ibox &bounds) const = 0;
	virtual void do_write(const r2::ipoint &topleft, const Image<> &img,
						  const r2::ibox &bounds_img) = 0;
	virtual void do_write(const r2::ibox &bounds, const color::radiance &c) = 0;
};

class default_large_image_impl : public large_image_impl
{
public:
	default_large_image_impl(std::unique_ptr<Image<>> img, 
							 const std::string &fname, 
							 const parameters &p, bool is_new);

	~default_large_image_impl();

private:
	std::unique_ptr<Image<>> m_image;
	bool m_dirty;
	std::string m_fname;
	parameters m_params;

	virtual std::unique_ptr<Image<>> do_read(const r2::ibox &bounds) const;
	virtual void do_write(const r2::ipoint &topleft, const Image<> &img,
						  const r2::ibox &bounds_img);
	virtual void do_write(const r2::ibox &bounds, const color::radiance &c);
};

template <>
class large_image<void>
	: public clonable, public movable
{
private:
	// order is important
	std::shared_ptr<large_image_impl> pimpl;
public:
	virtual ~large_image() {}

	size_t w() const { return size().w; }
	size_t h() const { return size().h; }
	const r2::usize &size() const { return pimpl->size(); }

	const image_traits<> &traits() const { return pimpl->traits(); }

	std::unique_ptr<Image<>> read(const r2::ibox &bounds) const;

	void write(const r2::ipoint &topleft, const Image<> &img);

	void write(const r2::ipoint &topleft, const Image<> &img,
			   const r2::ibox &bounds_img);

	void write(const r2::ibox &bounds, const color::radiance &c);

	DEFINE_MOVABLE(large_image);
	DEFINE_CLONABLE(large_image);
protected:
	large_image(std::unique_ptr<large_image_impl> _pimpl);
};

template <class I>
class large_image : public large_image<>
{
	static_assert(is_image<I>::value, "Must be an image");

	typedef large_image<> base;
public:
	large_image(std::unique_ptr<large_image_impl> pimpl) 
		: large_image<>(std::move(pimpl)) {}

	using large_image<>::write;
	I read(const r2::ibox &bounds) const;

	DEFINE_MOVABLE(large_image);
	DEFINE_CLONABLE(large_image);
};


template <class... PARAMS>
std::unique_ptr<large_image<>> load_large(const std::string &fname,
										  PARAMS &&...params);

template <class I, class...PARAMS> 
large_image<I> load_large(const std::string &fname, PARAMS &&...params);

template <class I, class...PARAMS> 
large_image<I> create_large(const r2::usize &s, const std::string &fname,
							PARAMS &&...);

r2::ibox bounds(const large_image<> &img);

}} // namespace s3d::img

#include "large_image.hpp"

#endif
