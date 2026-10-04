from scipy.special import ndtri


def normal_quantile(p):
    return float(ndtri(p))
