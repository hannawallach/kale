from collections import Counter
from numpy import asarray, cumsum, log, log2, exp, searchsorted, sqrt
from numpy.random import uniform
from scipy.spatial.distance import euclidean


def sample(dist, num_samples=1):
    """
    Uses the inverse CDF method to return samples drawn from the
    specified (unnormalized) discrete distribution.

    Arguments:

    dist -- (unnormalized) distribution

    Keyword arguments:

    num_samples -- number of samples to draw
    """

    cdf = cumsum(dist)
    r = uniform(size=num_samples) * cdf[-1]

    return cdf.searchsorted(r)


def log_sample(log_dist):

    return sample(exp(log_dist - log_dist.max()))


def log_sum_exp(x):
    """
    Returns log(sum(exp(x))).

    If the elements of x are log probabilities, they should not be
    exponentiated directly because of underflow. The ratio exp(x[i]) /
    exp(x[j]) = exp(x[i] - x[j]) is not susceptible to underflow,
    however. For any scalar m, log(sum(exp(x))) = log(sum(exp(x) *
    exp(m) / exp(m))) = log(sum(exp(x - m) * exp(m)) = log(exp(m) *
    sum(exp(x - m))) = m + log(sum(exp(x - m))). If m is some element
    of x, this expression involves only ratios of the form exp(x[i]) /
    exp(x[j]) as desired. Setting m = max(x) reduces underflow, while
    avoiding overflow: max(x) is shifted to zero, while all other
    elements of x remain negative, but less so than before. Even in
    the worst case scenario, where exp(x - max(x)) results in
    underflow for the other elements of x, max(x) will be
    returned. Since sum(exp(x)) is dominated by exp(max(x)), max(x) is
    a reasonable approximation to log(sum(exp(x))).
    """

    m = x.max()

    return m + log((exp(x - m)).sum())


def mean_relative_error(p, q, normalized=True):
    """
    Returns the mean relative error between a discrete distribution
    and some approximation to it.

    Arguments:

    p -- distribution
    q -- approximate distribution

    Keyword arguments:

    normalized -- whether the distributions are normalized
    """

    assert len(p) == len(q)

    p, q = asarray(p, dtype=float), asarray(q, dtype=float)

    if not normalized:
        p /= p.sum()
        q /= q.sum()

    return (abs(q - p) / p).mean()


def entropy(p, normalized=True):
    """
    Returns the entropy of a discrete distribution.

    Arguments:

    p -- distribution

    Keyword arguments:

    normalized -- whether the distribution is normalized
    """

    p = asarray(p, dtype=float)

    if not normalized:
        p /= p.sum()

    return -(p * log2(p)).sum()


def kl(p, q, normalized=True):
    """
    Returns the Kullback--Leibler divergence between a discrete
    distribution and some approximation to it.

    Arguments:

    p -- distribution
    q -- approximate distribution

    Keyword arguments:

    normalized -- whether the distributions are normalized
    """

    assert len(p) == len(q)

    p, q = asarray(p, dtype=float), asarray(q, dtype=float)

    if not normalized:
        p /= p.sum()
        q /= q.sum()

    return (p * log2(p / q)).sum()


def js(p, q, normalized=True):
    """
    Returns the Jensen--Shannon divergence (a form of symmetricized KL
    divergence) between two discrete distributions.

    Arguments:

    p -- first distribution
    q -- second distribution

    Keyword arguments:

    normalized -- whether the distributions are normalized
    """

    assert len(p) == len(q)

    p, q = asarray(p, dtype=float), asarray(q, dtype=float)

    if not normalized:
        p /= p.sum()
        q /= q.sum()

    m = 0.5 * (p + q)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def hellinger(p, q, normalized=True):
    """
    Returns the Hellinger distance between two discrete distributions.

    Arguments:

    p -- distribution
    q -- distribution

    Keyword arguments:

    normalized -- whether the distributions are normalized
    """

    assert len(p) == len(q)

    p, q = asarray(p, dtype=float), asarray(q, dtype=float)

    if not normalized:
        p /= p.sum()
        q /= q.sum()

    return euclidean(sqrt(p), sqrt(q)) / sqrt(2)


def vi(y, z):
    """
    Returns the variation of information (in bits) between two
    partitions (clusterings) of n data points. The maximum attainable
    value is log_2(n) bits. For example, vi(y=zeros(8, dtype=int),
    z=xrange(8)) will return a value of 3.0.

    y -- first partition
    z -- second partition
    """

    assert len(y) == len(z)

    D = 1.0 * len(y)

    vi = 0.0

    p_y = Counter(y)

    for i in p_y.keys():
        p_y[i] /= D
        vi -= p_y[i] * log2(p_y[i])

    p_z = Counter(z)

    for j in p_z.keys():
        p_z[j] /= D
        vi -= p_z[j] * log2(p_z[j])

    p_yz = Counter(zip(y, z))

    for i, j in p_yz.keys():
        p_yz[(i, j)] /= D
        vi -= (2 * p_yz[(i, j)] *
               log2(p_yz[(i, j)] / (p_y[i] * p_z[j])))

    return vi
