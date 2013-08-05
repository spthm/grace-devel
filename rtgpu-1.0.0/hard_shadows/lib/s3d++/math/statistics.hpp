namespace s3d { namespace math
{

template <class T, int D>
Matrix<T,D,D> covariance(const std::vector<Vector<T,D>> &samples)/*{{{*/
{
	auto u = mean(samples);

	Matrix<T,D,D> cov(0);

	for(size_t k=0; k<samples.size(); ++k)
	{
		for(size_t i=0; i<samples[k].size(); ++i)
		{
			for(size_t j=0; j<samples[k].size(); ++j)
				cov[i][j] += (samples[k][i] - u[i])*(samples[k][j] - u[j]);
		}
	}

	cov /= samples.size()-1;

	return cov;
}/*}}}*/

template <class T>
T variance(const std::vector<T> &samples)/*{{{*/
{
	auto u = mean(samples);

	T cov(0);

	for(size_t k=0; k<samples.size(); ++k)
		cov += (samples[k] - u)*(samples[k] - u);

	cov /= samples.size()-1;

	return cov;
}/*}}}*/

template <class T>
T mean(const std::vector<T> &samples)/*{{{*/
{
	assert(!samples.empty());

	T u(0); // mean

	for(size_t k=0; k<samples.size(); ++k)
		u += samples[k];
	u /= samples.size();

	return u;
}/*}}}*/

}} // namespace s3d::math
