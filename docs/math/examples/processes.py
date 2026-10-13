"""Exact stochastic transitions, generators, and generated irregular paths."""
import numpy as np
from convexity.math import processes as p, distributions as d


def main():
    rng = np.random.default_rng(130)
    ou = p.OrnsteinUhlenbeck(1.8, .5, .4)
    kernel = ou.transition(-1., .75)
    print('OU conditional mean/variance:', kernel.mean(), kernel.covariance())
    print('OU conditional quantiles:', kernel.distribution().quantile([.05,.5,.95]))
    print('OU generator of x² at x=1:', ou.generator(0, 1., 2., 2.))
    paths = ou.sample(-1., [0., .1, .4, 1., 2.], draws=5, rng=rng)
    print('Exact joint OU paths:', paths.values)

    vector = p.VectorOrnsteinUhlenbeck([[2., -.3], [.2, 1.]], [1., -1.], [[.4, .1], [.1, .3]])
    print('Vector OU stationary covariance:', vector.stationary().covariance())
    cir = p.CIR(2., .04, .5)
    print('CIR Feller condition:', cir.feller_satisfied)
    print('CIR exact law from boundary:', cir.transition(0., .5).distribution().quantile([.05,.5,.95]))
    print('GBM expected level:', p.GeometricBrownianMotion(.05,.2).transition(100.,1.).mean())

    jumps = p.AdditiveJumpDiffusion(.1, [.2], .8, d.normal(-.3,.1))
    print('Additive jump mean/variance:', jumps.transition(0.,1.).mean(), jumps.transition(0.,1.).covariance())
    merton = p.MertonJumpDiffusion(.05,.2,.8,-.1,.2).transition(100.,1.)
    samples = merton.sample(20000,rng=rng)
    print('Merton analytic/sample expected levels:', merton.mean(), samples.mean())


if __name__ == '__main__':
    main()
