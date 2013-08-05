#ifndef S3D_UTIL_ALGORITHM_H
#define S3D_UTIL_ALGORITHM_H

namespace s3d
{

template<class IN_IT, class OUT_IT, class PRED>
OUT_IT move_if(IN_IT itbeg, IN_IT itend, OUT_IT, itres, PRED pred)
{
	for (; itbeg != itend; ++itbeg)
	{
		if(pred(*itbeg))
		{
			*itres = std::move(*itbeg);
			++itres;
		}
	}
	return itres;
}

} // namespace s3d

#endif
