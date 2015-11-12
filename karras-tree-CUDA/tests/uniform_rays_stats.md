For a relatively brief description of the statistic tests used here to assess
the isotropy of the ray directions, see Fisher, Lewis and Embleton (1987):
section 5.3.1 discusses the Rayleigh test, and 5.6.1 covers An, Gn and Fn.
For a more in-depth discussion, see references therein and those below.

z is the Rayleigh Statistic, or Rayleigh's z-Test.
It tests the null hypothesis that there is no mean direction to the data.
That is, it tests for one-sidedness in the data. The null hypothesis is

    H0: The population is uniformly distributed

Further, this test assumes that the distribution is unimodal - that is, there
will only be one preferred direction at most. The z-Test will, for example, fail
to detect distribution of points which are clustered it two exactly-opposite
directions (i.e. diametrically bimodal data).

The statisic itself is,

    z = (p * R^2) / n,

where p is the dimension of the data, n is the number of directions, and R is
the resultant length of all direction vectors. For p = 3 (three dimensions),

    R^2 = [SUM(xi)]^2 + [SUM(yi)]^2 + [SUM(zi)]^2,

where each sum is over the n direction vectors, {x,y,z}i are the ith components
of the ith vector, and _each direction vector has length 1_. (We are testing
directionality, which is entirely separate from vector length; we _must_
normalize each vector before computing R^2.) We reject the null hypothesis for
large values of z.

More concretely, in three dimensions, the asymptotic (n -> infinity) null
distribution for z is

    z ~ (X_3)^2,

where (X_3)^2 is the Chi-squared distribution with 3 degrees of freedom (Diggle,
Fisher & Lee, 1985). This is appropriate for n >~ 10.

Tests of the Rayleigh Statistic are therefore quite straightforward: compute z,
and compare it to the degree-3 Chi-squared value at a chosen significance a.
Two common values of the significance level are 5% and 1%, with critical values
of z,

    P(z > 7.815) = 0.05
    P(z > 11.35) = 0.01

That is, if z < 7.815, there is no evidence to _reject_ the null hypothesis at
the 5% level.


An is Beran's Statistic.
It tests the null hypothesis that the distribution is uniform against
alternative models which are _not_ symmetric with respect to the centre of the
sphere,

    An =  n - 4 / (n * pi) * SUM(psi_ij),

where the sum is over _all pairs_ of direction vectors i != j, and psi_ij is
the angle between vectors i and j, defined as arccos((dot(v_i, v_j))) for
the pair of vectors i and j.

Gn in Gine's Statistic.
It tests the null hypothesis that the distribution is unfiform against
alternative models which _are_ symmetric with respect to the centre of the
sphere,

    Gn = (n / 2) - 4 / (n * pi) * SUM(sin(psi_ij))

where all terms are as for An.

Fn is also due to Gine, and tests the null hypothesis against all alternative
models,

    Fn = An + Gn.

Again, we reject the null hypothesis for too-large values of An, Gn or Fn.
The asymptotic distributions for the null hypothesis are infinite linear
combination of chi-squared variables (Diggle, Fisher & Lee, 1985). Some p-values
have been tabulated by Keilson et al. (1983), a small sample of which are given
below (note that they defined Y_Odd == An, Y_Even == Gn, Y == Fn),

    P(An > 1.414) = 0.2   P(An > 2.207) = 0.05   P(An > 3.090) = 0.01

    P(Gn > 0.646) = 0.2   P(Gn > 0.884) = 0.05   P(Gn > 1.135) = 0.01

    P(Fn > 1.948) = 0.2   P(Fn > 2.748) = 0.05   P(Fn > 3.633) = 0.01

So, for example, if Fn = 1.5, there is no evidence to _reject_ the null
hypothesis at the 20% level.



References
----------

Diggle, P. J., Fisher, N. I. & Lee, A. J. (1985),
"A comparison of tests of uniformity forspherical data",
Austral. J. Statist. 27, 53-59.

Fisher, N. I., Lewis, T. & Embleton, B. J. J. (1987),
"Statistical Analysis of Spherical Data",
Cambridge University Press, Online ISBN: 9780511623059.

Keilson, J., Petrondas, D., Sumita, U. & Wellner, J. (1983),
"Significance points for tests of uniformity on the sphere",
J. Statist. Comput. Simul 7, 195-218.
