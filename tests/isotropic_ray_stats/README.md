# Measures of ray isotropy

## Introduction

Described below, in brief, are the statistical tests used to assess the isotropy of
ray directions.


As is typical, these are framed for a hypothesis test where the null distribution is
isotropic.
While such tests can identify that a given distribution is __not__ isotropic, they __cannot__
identify the null hypothesis as being correct (for some reasonable definition of correct).
`ripleyk_stats.cu` and `uniformity_stats.cu` perform the former hypothesis test on a single
ray distribution, and are useful to ensure that ray distributions are not obviously-anisotropic.


In order to actually ascertain isotropy with some statistical rigour, we must instead
perform an equivalence test, or a noninferiority test.
(That is, a test of equivalence between the ray distributions and some other
distribution which is assumed isotropic.)
`perform_tests.py` does this; results for the Rayleigh test, `An` and `Gn` are
output to screen, and a plot of equivalence tests at several scales is saved to
`./plots` for Ripley's K-function.


Note that this Python script requires `ray_dump.cu` be compiled and run beforehand,
to produce samples of GRACE-generated rays (saved to `./ray_dumps`).
Note also that, when running from scratch with default arguments, it may take
`>= 24` hours to complete the test!

For a relatively brief description of some of the statistic tests used here to assess
isotropy, see [Fisher, Lewis and Embleton (1987)](#references); they cover the
Rayleigh test in Section 5.3.1, and the An and Gn statistics in Section 5.6.1.
For a more in depth discussion, consult the references provided below.


## K, Ripley's K-statistic

`K(s)` is Ripley's K-statistic, which is a test of spatial homogeneity at a scale
`s`.
It is defined as ([Dixon, 2002](#references))

```
    K(s) = (1 / lambda) * EXPECTED[N(s)]
```

where `lambda` is the number of points per unit area, `EXPECTED[x]` denotes the
expected value of `x`, and `N(s)` is the number of _other_ points within a distance
`s` of any given point.

For a set of `n` points, each with locations `r_i`, the unbiased estimator is,

```
    K'(s) = 1 / (lambda * (n - 1)) SUM[i != j] IDENTITY[|r_i - r_j| < s]
```

where the `SUM` is over _all pairs_ (i.e. `i = 1 to n`, `j = 1 to n`, but `i != j`),
`|x|` is the Euclidean norm of `x`, and `IDENTITY[c] = 1` if `c` is `True`, else `0`.
Under complete spatial randomness (CSR), we have

```
    K_CSR(s) = pi * s^2
```

i.e. the area within a distance `s`.

Here, the scale s is taken to be a great-circle distance on the unit sphere,
and `|r_i - r_j|` is then instead the great-circle distance, on the unit sphere,
between points `r_i` and `r_j`.
In this geometry, under CSR, `K_CSR(s)` is again the area (see also [Robeson et al., 2014](#references)),

```
    K_CSR(s) = 2pi * (1 - cos(s))
```

Positive values of `K'(s) - K_CSR(s)` indicate clustering on scale `s`, while
negative values indicate dispersion.


## W, Rayleigh's Statistic

W is the Rayleigh Statistic (sometimes Rayleigh's z-Test).
It tests the null null hypothesis that there is no mean direction to the data.
That is, it tests for one-sidedness in the data. Further, this test is not sensitive
to distributions which are symmetric with respect to the point of origin; it will,
for example, fail to detect distributions which are clustered it two exactly-opposite
directions (i.e. diametrically bimodal data).
It is most sensitive to unimodal distributions.

The statistic itself is,

```
    W = (p * R^2) / n,
```

where `p` is the dimension of the data, `n` is the number of directions, and `R`
is the resultant length of all direction vectors. For `p = 3` (3 dimensions),

```
    R^2 = [SUM(xi)]^2 + [SUM(yi)]^2 + [SUM(zi)]^2,
```

where each sum is over the `n` direction vectors, `{x,y,z}i` are the components
of the `i`th vector, and _each direction vector has length 1_. (We are testing
directionality, which is entirely separate from vector length; we _must_
normalize each vector before computing `R^2`.) We reject the null hypothesis for
large values of `W`.

More concretely, in three dimensions, the asymptotic (`n` -> infinity) null
distribution for z is

```
    W ~ (X_3)^2,
```

where `(X_3)^2` is the Chi-squared distribution with 3 degrees of freedom
([Diggle, Fisher & Lee, 1985](#references)). This is appropriate for `n >~ 10`.

Tests of the Rayleigh Statistic are therefore quite straightforward: compute W,
and compare it to the degree-3 Chi-squared value at a chosen significance level.
Two common values of the significance level are 5% and 1%, with critical values
of W,

    P(W > 7.815) = 0.05
    P(W > 11.35) = 0.01

That is, if `W > 7.815`, the null hypothesis is rejected at the 5% level.


## An and Gn

`An` is Beran's Statistic.
It tests the null hypothesis that the distribution is uniform against alternative
models which _are not_ symmetric with respect to the centre of the sphere

    An =  n - 4 / (n * pi) * SUM[i < j <= n] (psi_ij),

where the sum is over _all distinct pairs_ of direction, and `psi_ij` is
the angle between vectors `i` and `j`, defined as `arccos[dot(v_i, v_j)]` for
the pair of vectors `i` and `j`.

`Gn` in Gine's Statistic.
It tests the null hypothesis that the distribution is uniform against alternative
models which _are_ symmetric with respect to the centre of the sphere,

    Gn = (n / 2) - 4 / (n * pi) * SUM[i < j <= n] (sin(psi_ij))

where all terms are as for `An`.

We reject the null hypothesis for too-large values of `An` or `Gn`.
The asymptotic distributions for the null hypothesis are infinite linear combination
of chi-squared variables ([Diggle, Fisher & Lee, 1985](#references)). Some p-values
have been tabulated by [Keilson et al. (1983)](#references), a small sample of which are given
below (note that they defined `Y_Odd == An`, `Y_Even == Gn`),

    P(An > 2.207) = 0.05   P(An > 3.090) = 0.01

    P(Gn > 0.884) = 0.05   P(Gn > 1.135) = 0.01


## References

Diggle, P. J., Fisher, N. I. & Lee, A. J. (1985),
"A comparison of tests of uniformity for spherical data",
Austral. J. Statist. 27, 53-59.

Fisher, N. I., Lewis, T. & Embleton, B. J. J. (1987),
"Statistical Analysis of Spherical Data",
Cambridge University Press, Online ISBN: 9780511623059.

Keilson, J., Petrondas, D., Sumita, U. & Wellner, J. (1983),
"Significance points for tests of uniformity on the sphere",
J. Statist. Comput. Simul 7, 195-218.

Dixon, P. M. (2002),
"Ripley’s K function",
Encycl. Environmetrics, Volume 3, 1796–1803.
John Wiley & Sons Ltd., Chichester.

Robeson, S. M., Li, A., and Huang, C. (2014),
"Point-pattern analysis on the sphere",
Spat. Stat., Volume 10, 76–86.
