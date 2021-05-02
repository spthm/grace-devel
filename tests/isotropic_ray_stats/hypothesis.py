from __future__ import print_function

import numpy as np
import scipy.stats as sstat

def _indicator(x, y):
    return int(0.5 * (np.sign(x - y) + 1))

def _z_score(cl):
    return sstat.norm.ppf(cl)

def _t_score(cl, dof):
    return sstat.t.ppf(cl, df=dof)

def _sample_variance(X):
    return np.var(X, ddof=1)

def _welch_satterthwaite_dof(s2x, s2y, nx, ny):
    """
    Return estimate for degrees of freedom for sample variances s2x and s2y.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        dof = (s2x / nx + s2y / ny)**2 / ( s2x*s2x / (nx*nx*(nx-1)) +
                                           s2y*s2y / (ny*ny*(ny-1)) )
    if np.isnan(dof):
        # All variances must have been zero, or very close. So return something
        # maybe vaguely sensible.
        dof = (nx + ny) / 2.
        raise RuntimeWarning("Warning: Welch Satterthwaite DoF is Nan")
    return dof

def _standard_error_mean_difference(X, Y, ret_var=False):
    """
    Return standard error on the difference of the means of X and Y.

    If ret_var is true, the sample variances of X and Y are also returned.
    """
    # The standard error from Welsh's t-test; we do not assume equal population
    # variances of X and Y since, in equivalence testing, our null (assumed)
    # hypothesis is that X and Y are drawn from different distributions.
    # (The alternative, when variances are equal, is the pooled variance.)
    # See also Welch-Satterthwaite equation.
    s2x = _sample_variance(X)
    s2y = _sample_variance(Y)
    se = np.sqrt(s2x / len(X) + s2y / len(Y))
    if ret_var:
        return se, s2x, s2y
    return se

def _mean_diff_confidence_interval(X, Y, cl):
    """
    Return the confidence interval for the difference in the means of X and Y.

    cl is the desired confidence level of the interval.

    low, high are returned, where the (100*cl)% confidence interval for the
    difference is [low, high].
    """
    mean_diff = np.mean(X) - np.mean(Y)
    SE, s2x, s2y = _standard_error_mean_difference(X, Y, ret_var=True)
    dof = _welch_satterthwaite_dof(s2x, s2y, len(X), len(Y))
    t = _t_score(cl, dof)
    # The min, max here are necessary for strict conformance to a type-I error
    # rate of 1 - cl (a type I error is a rejection of a true null hypothesis).
    # Though for symmetric alpha, as used here, the standard (100*cl)%
    # confidence interval --- which is identical to the below, but without the
    # min/max --- is fine.
    # See e.g. Berger, Roger L.; Hsu, Jason C.; "Bioequivalence trials,
    # intersection-union tests and equivalence confidence sets."; Statistical
    # Science, 11(4) (1996), pp. 283--319.
    # doi:10.1214/ss/1032280304
    low = min(0, mean_diff - t * SE)
    high = max(0, mean_diff + t * SE)

    return low, high

def _mann_whitney(X, Y):
    """
    Return Mann-Whitney estimator and estimated std. dev. wxy, sxy for X, Y.

    wxy is the MW-estimator for P[X > Y] and sxy is the square root of the
    varaince estimator of wxy.
    """
    m = len(X)
    n = len(Y)

    wxy = 0
    for i in range(m):
        for j in range(n):
            wxy += _indicator(X[i], Y[j])
    wxy /= float(m * n)

    wxxy = 0
    for i1 in range(m):
        for i2 in range(i1 + 1, m):
            for j in range(n):
                wxxy += _indicator(min(X[i1], X[i2]), Y[j])
    wxxy *= 2. / (m * (m - 1) * n)

    wxyy = 0
    for i in range(m):
        for j1 in range(n):
            for j2 in range(j1 + 1, n):
                wxyy += _indicator(X[i], max(Y[j1], Y[j2]))
    wxyy *= 2. / (n * (n - 1) * m)

    sxy = wxy - (m + n - 1)*wxy*wxy + (m - 1)*wxxy + (n - 1)*wxyy
    sxy *= 1. / (m * n)
    sxy = np.sqrt(sxy)

    return wxy, sxy

def equivalence_test(X, Y, e1, e2, cl=0.95):
    """
    Perform a TOST equivalence test for normally-distributed X, Y.

    Return (result, low, high) where result is True if the null hypothesis of
    different distributions is rejected, i.e. if X =~ Y, else False; low and
    high are the lower and upper (100*cl)% confidence bounds, respectively.

    result will be True iff low > -e1 and high < e2. Note that if low <= -e1
    and high >= e2, we fail to reject the null hypothesis, but the result is
    inconclusive (the test does not have sufficient statistical power).

    X and Y are the (normal) distributions to compare.

    e1 and e2 are the indifference intervals, defining the region of similarity,
    which must be selected appropriately for the problem at hand. Both must be
    positive. They are in the same units as the values in X and Y.

    cl is the confidence level of the returned interval (must be in (0, 1)).
    """
    if e1 < 0 or e2 < 0:
        raise ValueError("e1 and e2 must be non-negative")
    if cl <= 0.0 or cl >= 1.0:
        raise ValueError("cl must lie in the open interval (0.0, 1.0)")

    low, high = _mean_diff_confidence_interval(X, Y, cl)
    reject = low > -e1 and high < e2

    return reject, low, high

def noninferiority_test(X, Y, e, cl=0.95, inferior='larger'):
    """
    Perform a noninferiority test for normally distributed X and Y.

    Return (result, low, high) where result is True if the null hypothesis of
    a noninferior X distribution is rejected, else False; low and high are the
    lower and upper (100*cl)% confidence bounds, respectively.

    inferior defines the direction of inferiority: 'larger' produces a test that
    X is not significantly larger than Y; 'smaller' produces a test that X is
    not significantly smaller than Y.

    For the former (larger is inferior), result will be True iff
        high < e.
    For the latter (smaller is inferior), result will be True iff
        low > -e.

    e is the indifference range, defining the region of noninferiority, and
    must be selected appropriately for the problem at hand. It must be positive.
    It is in the same units as teh values in X and Y.

    cl is the confidence level of the returned result (must be in (0, 1)).
    """
    if e < 0:
        raise ValueError("e must be non-negative")
    if cl <= 0.0 or cl >= 1.0:
        raise ValueError("cl must lie in the open interval (0.0, 1.0)")
    if inferior not in ['larger', 'smaller']:
        raise ValueError("inferior must be one of 'larger', 'smaller'")

    low, high = _mean_diff_confidence_interval(X, Y, cl)

    if inferior == 'larger':
        reject = high < e
    else:
        reject = -e < low

    return reject, low, high

def nonnormal_equivalence_test(X, Y, e1=0.1, e2=0.1, cl=0.95):
    """
    Perform a Mann-Whitney equivalence test for non-normally distributed X, Y.

    Return (result, wxy, sxy, test_stat, C) where result is True if the null
    hypothesis of different distributions is rejected, i.e. if X =~ Y, else
    False; wxy is the Mann-Whitney estimator for P[X > Y]; sxy is the estimator
    of the standard deviation of wxy; test_stat is the actual test statistic;
    and C is the critical value for rejection.

    result will be True iff test_stat < C

    X and Y are the distributions to compare.

    e1 and e2 are the indifference intervals, defining the region of similarity,
    which must be selected appropriately for the problem at hand. Both must be
    positive. Default values of e1 = e2 = 0.10 is a relatively standard, strict
    condition.

    cl is the confidence level of the returned result (must be in (0, 1)).
    """
    if e1 < 0 or e2 < 0:
        raise ValueError("e1 and e2 must be non-negative")
    if cl <= 0.0 or cl >= 1.0:
        raise ValueError("cl must lie in the open interval (0.0, 1.0)")

    wxy, sxy = _mann_whitney(X, Y)

    rootnc = (e1 + e2) / (2. * sxy)
    nc = rootnc * rootnc
    DoF = 1
    C = np.sqrt(sstat.ncx2.ppf(1 - cl, DoF, nc))

    delta = 0.5 + (e2 - e1) / 2.0
    test_stat = abs(wxy - delta) / sxy
    reject = test_stat < C

    return reject, wxy, sxy, test_stat, C

def nonnormal_noninferiority_test(X, Y, e=0.1, cl=0.95, inferior='larger'):
    """
    Perform a Mann-Whitney noninferiority test for non-normal X and Y.

    Return (result, wxy, sxy, test_stat, C) where result is True if the null
    hypothesis of different distributions is rejected, i.e. if X =~ Y, else
    False; wxy is the Mann-Whitney estimator for P[X > Y]; sxy is the estimator
    of the standard deviation of wxy; test_stat is the actual test statistic;
    and C is the critical value for rejection, here equal to the cl(th)-quantile
    of the standard normal distribution.

    inferior defines the direction of inferiority: 'larger' produces a test that
    X is not significantly larger than Y; 'smaller' produces a test that X is
    not significantly smaller than Y.

    result will be True iff
        test_stat > C;
    where for both inferior being 'larger' and 'smaller', for equal confidence
    limit cl, C is equal, and for identical X, Y and e, abs(test_stat) is
    equal.

    X and Y are the distributions to compare.

    e is the indifference range, defining the region of noninferiority, and
    must be selected appropriately for the problem at hand. It must be positive.
    A default value of e 0.10 is a relatively standard, strict condition.

    cl is the confidence level of the returned result (must be in (0, 1)).
    """
    if e < 0:
        raise ValueError("e must be non-negative")
    if cl <= 0.0 or cl >= 1.0:
        raise ValueError("cl must lie in the open interval (0.0, 1.0)")
    if inferior not in ['larger', 'smaller']:
        raise ValueError("inferior must be one of 'larger', 'smaller'")

    wxy, sxy = _mann_whitney(X, Y)
    C = sstat.norm.ppf(cl)

    if inferior == 'larger':
        test_stat = ((0.5 + e) - wxy) / sxy
        reject =  test_stat > C
    else:
        test_stat = (wxy - (0.5 - e)) / sxy
        reject = test_stat > C

    return reject, wxy, sxy, test_stat, C

def _print_mean_std(X, Y):
    print("E(X): %.1f, std(X): %.1f\nE(Y): %.1f, std(Y): %.1f"
          % (np.mean(X), np.std(X, ddof=1), np.mean(Y), np.std(Y, ddof=1)))

def _print_conf_equiv(low, high, e1, e2, equiv):
    print("  95%% confidence interval: [%.2f, %.2f]" % (low, high))
    print("  Zone of equivalence:     [%.2f, %.2f]" % (-e1, e2))
    result = "equivalence" if equiv else "fail to reject null"
    print("  Result: " + result)

def _print_conf_noninf(limit, e, noninf, inferior='larger'):
    if noninf == 'larger':
        print("  95%% upper confidence limit: %.2f" % (limit,))
        print("  Zone of noninferiority:    [-inf, %.2f]" % (abs(e), ))
        result = "noninferiority" if noninf else "fail to reject null"
    else:
        print("  95%% lower confidence limit: %.2f" % (limit,))
        print("  Zone of noninferiority:    [%.2f, +inf]" % (-abs(e), ))
        result = "noninferiority" if noninf else "fail to reject null"
    print("  Result: " + result)

def _print_crit_equiv(test_stat, crit, equiv):
    print("  Test statistic:     %.4f" % (test_stat, ))
    print("  95%% critical value: %.4f" % (crit, ))
    result = "equivalence" if equiv else "fail to reject null"
    print("  Result: " + result)

def _check_close_interval(low, high, true_low, true_high, desc=None,
                          **np_kwargs):
    if desc is None:
        desc = ""

    if np.isclose(low, true_low, **np_kwargs) and \
       np.isclose(high, true_high, **np_kwargs):
        print("PASSED %s" % (desc, ))
        return True
    else:
        print("FAILED %s:" % (desc, ))
        print("  Actual confidence interval: [%.2f, %.2f]"
              % (true_low, true_high))
        return False

def _check_close_limit(limit, true_limit, desc=None, **np_kwargs):
    if desc is None:
        desc = ""

    if np.isclose(limit, true_limit, **np_kwargs):
        print("PASSED %s" % (desc, ))
        return True
    else:
        print("FAILED %s:" % (desc, ))
        print("  Actual confidence limit: %.2f" % (true_limit, ))
        return False

def _check_close_multiple(observed, actual, desc=None, val_descs=None,
                          **np_kwargs):
    if len(observed) != len(actual):
        raise ValueError("Observed and actual value list lengths no equal")
    if val_descs is None:
        val_descs = [None] * len(observed)
    elif len(val_descs) != len(observed):
        raise ValueError("Value descriptions and observed values list length not consistent")

    if desc is None:
        desc = ""

    for o, a, d in zip(observed, actual, val_descs):
        if not np.isclose(o, a, **np_kwargs):
            print("FAILED %s:" %(d, ))
            print("  Actual value: %.4f" % (a, ))
            return False
    print("PASSED %s" % (desc, ))

if __name__ == '__main__':
    # Test data consistent with example at:
    # https://onlinecourses.science.psu.edu/stat509/node/55
    # Accessed 2017-02-22.
    # normal, mean ~ 17.4, sample std. dev. ~ 6.5
    X = np.array([10.32288148, 18.47990715, 25.45856056, 20.23516944,
                  26.52574269,  9.74212874, 14.69967343,  9.49407793,
                  18.86892608,  5.34457132, 13.60168844, 13.52668511,
                  14.85206486, 19.6427994 , 10.36880158,  4.52797006,
                  16.08109902, 12.90180413, 25.78748175, 28.5665357 ,
                  12.6323904 , 23.09875378, 16.09770611, 23.92229129,
                  21.93078951, 26.08576392, 21.46189056, 20.0336223 ,
                  23.60204535, 14.09647028])
    # normal, mean ~ 20.6, sample std. dev. ~ 6.5
    Y = np.array([ 9.14252572, 19.38203127, 20.97039489, 20.20088713,
                  20.20803573,  7.42781738, 18.45799345, 26.3898127 ,
                  20.7269037 , 16.62582436, 26.82100568, 14.64004008,
                  12.31144577, 17.72535396, 19.2877205 , 29.75264772,
                  20.30141356, 33.97349748, 24.53414498, 30.10515467,
                  11.63447084, 23.41704018, 25.44240116, 27.76956435,
                  11.5825022 , 25.63466745, 23.26837671, 20.88892028,
                  14.00730522, 25.35244])
    e = 4

    equiv, low, high = equivalence_test(X, Y, e, e, 0.95)
    _print_mean_std(X, Y)
    _print_conf_equiv(low, high, e, e, equiv)
    # Confidence interval should be ~[6.0, 0.0]
    passed = _check_close_interval(low, high, -6.0, 0.0,
                                   desc="normal X, Y equivalence test",
                                   rtol=1e-3, atol=1e-5)
    print()

    noninf, low, _ = noninferiority_test(X, Y, e, 0.95, inferior='smaller')
    _print_mean_std(X, Y)
    _print_conf_noninf(low, e, noninf, inferior='smaller')
    # Lower confidence limit should be ~6.0.
    passed = _check_close_limit(low, -6.0,
                                desc="normal X, Y noninferiority test",
                                rtol=1e-3)
    print()


    # Test data from Table 6.3 of "Testing Statistical Hypotheses of Equivalence
    # and Noninferiority, Second Edition by Stefan Wellek (2010), pp. 123
    # ISBN: 978-1439808184
    X = np.array([10.3, 11.3, 2.0, -6.1, 6.2, 6.8, 3.7, -3.3, -3.6, -3.5, 13.7,
                  12.6])
    Y = np.array([3.3, 17.7, 6.7, 11.1, -5.8, 6.9, 5.8, 3.0, 6.0, 3.5, 18.7,
                  9.6])
    e1 = 0.1382
    e2 = 0.2602

    equiv, wxy, sxy, test_stat, crit = nonnormal_equivalence_test(X, Y, e1, e2,
                                                                  0.95)
    _print_mean_std(X, Y)
    _print_crit_equiv(test_stat, crit, equiv)
    # Actual values from above Wellek (2010), Section 6.2, pp. 128.
    passed = _check_close_multiple([wxy, sxy, test_stat, crit],
                                   [0.41667, 0.11133, 1.2964, 0.30078],
                                   desc="Mann-Whitney X, Y equivalence test",
                                   val_descs=["MW stat.", "MW std. dev.",
                                              "MW test stat.",
                                              "MW critical value"],
                                   atol=1e-6, rtol=1e-4)
    print()
