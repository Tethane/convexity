"""Build and optionally execute the eleven financial notebooks on local inputs.

Run with the intended notebook interpreter:
    python scripts/build_financial_notebooks.py --execute
"""

from pathlib import Path
import argparse
import sys
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = """
from pathlib import Path
import sys
root = Path.cwd()
if not (root / 'convexity').exists():
    root = root.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from convexity.math import contracts as ct, curves as cv, pricing as pr
rng = np.random.default_rng(SEED)
ref, expiry = '2026-01-01', '2027-01-01'
discount, dividend = cv.flat_curve(ref, .04), cv.flat_curve(ref, .01)
model = pr.BlackScholesModel(100., discount, dividend, .2)
plt.rcParams.update({'figure.figsize': (9, 3.5), 'figure.dpi': 110,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.alpha': .2})
print('Interpreter:', sys.executable)
"""


def clean(s):
    return textwrap.dedent(s).strip()


def notebook(title, seed, intro, sections):
    cells = [
        nbf.v4.new_markdown_cell("# " + title + "\n\n" + clean(intro)),
        nbf.v4.new_code_cell(clean(SETUP).replace("SEED", str(seed))),
    ]
    for heading, narrative, code in sections:
        cells.extend(
            [
                nbf.v4.new_markdown_cell("## " + heading + "\n\n" + clean(narrative)),
                nbf.v4.new_code_cell(clean(code)),
            ]
        )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Scope of the evidence\n\nThese are generated-data experiments under the stated models, "
            "not market forecasts or universal model rankings. Numerical errors, backend restrictions "
            "and measure assumptions are documented in the "
            "[financial catalog](../docs/financial_mathematics.md) and "
            "[QA record](../docs/financial_qa.md). Automated adversarial checks live in `tests/math/finance/`."
        )
    )
    return nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "convexity": {
                "family": seed // 100,
                "data": "generated",
                "generator": "scripts/build_financial_notebooks.py",
            },
        },
    )


BOOKS = {}
BOOKS["measures"] = notebook(
    "P and Q: laws, likelihoods and valuation",
    1600,
    """Family 16. The physical drift and pricing drift are supplied independently.
Reweighting has meaning only because the two specified GBM laws share diffusion
coefficients. Time is in years; rates and volatilities are decimals.""",
    [
        (
            "Change a drift, preserve diffusion",
            """The market price of Brownian risk links these two model specifications.
The same physical Brownian draws support a likelihood-ratio comparison with the
analytic Q expectation. Finite-sample normalization changes the estimator.""",
            """
from convexity.math import measures as me, simulation as sim
pair = me.gbm_pair(.12,.04,.01,.2)
lam = pair.market_price_of_risk(0.,100.)[0]
z = rng.normal(size=60000)
terminal = 100*np.exp(.12-.5*.2**2+.2*z)
log_density = me.girsanov_log_density(z[:,None,None], [lam], [1.])
paths = sim.PathScenarios([0.,1.],np.c_[np.full(len(z),100.),terminal],sampling='iid')
q_paths = me.reweight(paths,log_density)
display(pd.DataFrame({'quantity':['P mean','Q analytic mean','Q self-normalized mean','Mean dQ/dP','Weight effective size'],
                     'value':[terminal.mean(),100*np.exp(.03),q_paths.mean()[-1],np.exp(log_density).mean(),q_paths.effective_size]}))
fig,ax=plt.subplots()
ax.hist(terminal,bins=70,density=True,alpha=.45,label='Physical draws')
ax.hist(terminal,bins=70,weights=q_paths.weights,density=True,alpha=.45,label='Q reweighted law')
ax.set(xlabel='Terminal spot',ylabel='Density',title=f'Same volatility, different drift; lambda={lam:.2f}')
ax.legend();plt.show()
""",
        ),
        (
            "Weight concentration as the drift gap grows",
            """Weight concentration measures the difficulty of importance sampling. It is not
the number of independent observations after reweighting, and it does not justify
an IID standard error.""",
            """
rows=[]
for mu in np.linspace(.03,.6,25):
    risk=(mu-.03)/.2
    weights=me.reweight(paths,me.girsanov_log_density(z[:,None,None],[risk],[1.]))
    rows.append((mu,risk,weights.effective_size/len(z)))
concentration=pd.DataFrame(rows,columns=['P drift','Risk premium','Effective size fraction'])
concentration.plot(x='P drift',y='Effective size fraction',legend=False)
plt.ylabel('1 / sum(w²), divided by draw count');plt.title('Importance weights concentrate');plt.show()
""",
        ),
        (
            "Discounted Q payoffs and explicit composition",
            """Pricing uses the supplied Q numeraire. The reweighted finite law supplies a
price estimate without an automatic Monte Carlo standard error. The analytic
price is an independent QuantLib reference.""",
            """
q_model=pr.BlackScholesModel.from_measure_pair(pair,100.,ref)
contract=ct.VanillaContract(100.,ct.Exercise('european',(expiry,)))
analytic=pr.price(pr.PricingProblem(contract,q_model,ref),pr.VanillaEngine())
scenarios=pr.scenario_price(q_paths,lambda path:max(path[-1]-100.,0.),pair.pricing)
display(pd.DataFrame({'method':['QuantLib analytic','Reweighted P scenarios'],
                     'value':[analytic.value,scenarios.value],
                     'standard error':[analytic.error_estimate,scenarios.error_estimate]}))
t=np.linspace(0,5,50)
plt.plot(t,me.MoneyMarketNumeraire(.04).value(t),label='Numeraire N(t)')
plt.plot(t,np.exp(-.04*t),label='Discount 1/N(t)');plt.xlabel('Years');plt.legend();plt.show()
""",
        ),
    ],
)

BOOKS["contracts"] = notebook(
    "Contracts: cash flows, conventions and settlement",
    1700,
    """Family 17. All cash flows and prices are supplied or generated. A contract's
multiplier and settlement convention affect delivery quantities; an FX conversion
factor is supplied explicitly and is not an FX forecast.""",
    [
        (
            "Coupon schedule and day-count effects",
            """The stub schedule makes the accrual convention visible. Actual/365 and 30/360
give different coupon amounts for the same annual coupon rate.""",
            """
schedule=ct.schedule('2026-02-10','2028-01-01',months=6)
rows=[]
for convention in ['actual365','actual360','30_360']:
    for cf in ct.fixed_cashflows(schedule,.05,convention=convention):
        rows.append((ct.to_date(cf.date).isoformat(),convention,cf.amount))
flows=pd.DataFrame(rows,columns=['Payment date','Day count','Amount'])
display(flows)
flows.groupby(['Payment date','Day count']).Amount.sum().unstack().plot.bar()
plt.ylabel('Cash per 100 principal');plt.title('Stub coupons and final redemption');plt.tight_layout();plt.show()
""",
        ),
        (
            "Cash versus physical settlement",
            """Physical delivery and its cash leg reconcile to cash exercise value at the
same spot. The table makes the sign of put delivery explicit.""",
            """
spots=np.linspace(60,140,100)
rows=[]
for kind in ['call','put']:
    contract=ct.VanillaContract(100.,ct.Exercise('european',(expiry,)),kind,100.,'physical')
    deliveries=[contract.delivery(s) for s in spots]
    value=np.array([units*s+cash for s,(units,cash) in zip(spots,deliveries)])
    plt.plot(spots,value,label=f'{kind}: delivered position marked at spot')
    for spot in [80.,100.,120.]:
        units,cash=contract.delivery(spot)
        rows.append((kind,spot,units,cash,units*spot+cash))
plt.xlabel('Exercise spot');plt.ylabel('Contract exercise value');plt.legend();plt.show()
display(pd.DataFrame(rows,columns=['Kind','Spot','Delivered shares','Cash leg','Value']))
""",
        ),
        (
            "Corporate actions and conversion conventions",
            """The split and ex-dividend transformations conserve marked wealth under the
specified frictionless assumptions. Compounding frequency changes the accumulation
factor even when the quoted annual rate is unchanged.""",
            """
q,p=ct.split_holdings(10.,100.,4.)
qd,pd_,cash=ct.dividend_adjustment(10.,100.,2.)
display(pd.DataFrame({'case':['Before','4-for-1 split','Ex-dividend'],
 'shares':[10,q,qd],'price':[100,p,pd_],'cash':[0,0,cash],'wealth':[1000,q*p,qd*pd_+cash]}))
periods=np.linspace(0,10,101)
for convention in ['simple','periodic','continuous']:
    plt.plot(periods,[ct.interest_factor(.05,t,compounding=convention) for t in periods],label=convention)
plt.xlabel('Years');plt.ylabel('Accumulation factor');plt.legend();plt.show()
print('PV of supplied EUR cash flow at 1.10 USD/EUR:',ct.present_value([ct.CashFlow(expiry,100.,'EUR')],discount,fx={'EUR':1.10}))
""",
        ),
    ],
)

BOOKS["curves"] = notebook(
    "Curves: bootstrapping, forwards and carry",
    1800,
    """Family 18. Interest rates below are synthetic decimal quotes. Discount factors
are date-based, zero/forward rates are continuously compounded, and curve
extrapolation is deliberately disabled.""",
    [
        (
            "A shaped zero curve",
            """Zero rates and interval forwards encode different information. A forward rate
is calculated from two discounts; it is not a forecast of the future realized rate.""",
            """
dates=[ref,'2027-01-01','2028-01-01','2029-01-01','2031-01-01']
curve=cv.zero_curve(dates,[.02,.028,.035,.039,.042])
times=np.linspace(.05,4.9,80)
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
axes[0].plot(times,[curve.discount(t) for t in times]);axes[0].set(xlabel='Years',ylabel='Discount factor')
axes[1].plot(times,[curve.zero_rate(t) for t in times],label='Zero')
axes[1].plot(times,[curve.forward_rate(t,t+.05) for t in times],label='0.05-year forward')
axes[1].set(xlabel='Years',ylabel='Annual decimal rate');axes[1].legend();plt.show()
""",
        ),
        (
            "Reprice bootstrap quotes without vendor helpers",
            """Quotes use explicit day counts and settlement conventions. The returned curve
and its quote residuals are calibrated snapshots. Convexity manages native
helpers and valuation dates internally.""",
            """
quotes=[(1,.025),(3,.027),(6,.03),(12,.034)]
helpers=[cv.deposit_quote(rate,f"{months}M") for months,rate in quotes]
boot=cv.bootstrap(ref,helpers)
comparison=pd.DataFrame({'Months':[x[0] for x in quotes],'Quote':[x[1] for x in quotes],
                         'Repriced':np.array([x[1] for x in quotes])+boot.calibration_errors})
comparison['Residual']=comparison.Repriced-comparison.Quote
display(comparison)
grid=np.linspace(.01,1,100)
plt.plot(grid,[boot.discount(t) for t in grid]);plt.xlabel('Years');plt.ylabel('Discount factor');plt.title('Bootstrapped deposit curve');plt.show()
""",
        ),
        (
            "Separate financing, dividend and storage assumptions",
            """Carry uses deterministic curves. Futures basis is the difference from a
separately supplied futures price; this calculation does not assume a stochastic
rate convexity adjustment.""",
            """
maturities=np.linspace(0,5,80)
for yield_ in [0.,.02,.05]:
    div=cv.flat_curve(ref,yield_)
    plt.plot(maturities,[cv.forward_price(100,t,discount,dividend=div) for t in maturities],label=f'Dividend yield {yield_:.0%}')
plt.xlabel('Years');plt.ylabel('Forward price');plt.legend();plt.show()
display(pd.DataFrame([{'Storage rate':u,'Convenience yield':y,'One-year forward':cv.forward_price(100,1,discount,storage_rate=u,convenience_yield=y)} for u,y in [(0,0),(.03,0),(.03,.02)]]))
""",
        ),
    ],
)

BOOKS["pricing"] = notebook(
    "Pricing engines: identity, convergence and capability",
    1900,
    """Family 19. QuantLib performs pricing. We compare engines under an identical
constant-coefficient model, then compare separately declared path-dependent and
multi-asset capabilities. All prices are per unit multiplier.""",
    [
        (
            "Put-call parity and quote Greeks",
            """Parity is an independent financial identity. Greeks below hold the model's
other quoted inputs fixed and are per unit change, not per percentage point.""",
            """
from dataclasses import replace
strikes=np.linspace(70,130,31)
rows=[]
for strike in strikes:
    vals=[]
    for kind in ['call','put']:
        contract=ct.VanillaContract(strike,ct.Exercise('european',(expiry,)),kind)
        vals.append(pr.price(pr.PricingProblem(contract,model,ref),pr.VanillaEngine()))
    rows.append((strike,vals[0].value,vals[1].value,vals[0].value-vals[1].value-(100*np.exp(-.01)-strike*np.exp(-.04)),vals[0].greeks['delta'],vals[0].greeks['vega']))
prices=pd.DataFrame(rows,columns=['Strike','Call','Put','Parity residual','Call delta','Call vega'])
prices.plot(x='Strike',y=['Call','Put']);plt.ylabel('Price');plt.show();display(prices.iloc[::5])
""",
        ),
        (
            "Grid convergence versus Monte Carlo uncertainty",
            """Tree and FD errors are discretization errors. Monte Carlo standard errors
measure sampling variability, not bias. A binomial engine is only admitted here
for constant coefficients; shaped curves require a suitable engine such as FD.""",
            """
contract=ct.VanillaContract(100.,ct.Exercise('european',(expiry,)))
problem=pr.PricingProblem(contract,model,ref)
reference=pr.price(problem,pr.VanillaEngine()).value
rows=[]
for steps in [25,50,100,200,400]:
    for method in ['tree','fd']:
        value=pr.price(problem,pr.VanillaEngine(method,steps=steps,grid=max(50,steps))).value
        rows.append((method,steps,value,value-reference))
convergence=pd.DataFrame(rows,columns=['Engine','Steps','Price','Error'])
for name,data in convergence.groupby('Engine'):
    plt.plot(data.Steps,data.Error,marker='o',label=name)
plt.axhline(0,color='black',lw=.8);plt.xscale('log');plt.xlabel('Grid steps');plt.ylabel('Error against analytic');plt.legend();plt.show()
mc=[]
for n in [2000,8000,32000]:
    result=pr.price(problem,pr.VanillaEngine('mc',steps=2,samples=n,seed=71))
    mc.append((n,result.value,result.error_estimate))
display(pd.DataFrame(mc,columns=['Draws','MC value','Standard error']))
""",
        ),
        (
            "Heston, averages and correlation",
            """These are distinct claims and models. Basket correlation changes the law of a
weighted sum. Arithmetic and geometric Asian values differ because their payoffs
differ; they are not two estimators of the same price.""",
            """
heston=pr.HestonModel(100,discount,dividend,.04,2.,.04,.3,-.7)
values=[pr.price(pr.PricingProblem(ct.VanillaContract(k,ct.Exercise('european',(expiry,))),heston,ref),pr.VanillaEngine('heston')).value for k in strikes]
plt.plot(strikes,prices.Call,label='BSM');plt.plot(strikes,values,label='Heston');plt.xlabel('Strike');plt.ylabel('Call price');plt.legend();plt.show()
fixings=['2026-04-01','2026-07-01','2026-10-01',expiry]
claims=[('Geometric Asian',pr.asian_price(model,ref,100,fixings,average='geometric')),
        ('Arithmetic Asian',pr.asian_price(model,ref,100,fixings,samples=12000))]
for rho in [-.5,0.,.8]:
    claims.append((f'Basket correlation {rho}',pr.basket_price([model,model],[[1,rho],[rho,1]],[.5,.5],ref,expiry,100,samples=12000)))
display(pd.DataFrame([(name,r.value,r.error_estimate) for name,r in claims],columns=['Claim','Value','Standard error']))
""",
        ),
    ],
)

BOOKS["implied_volatility"] = notebook(
    "Implied volatility: inversion, shape and arbitrage",
    2000,
    """Family 20. Implied volatilities invert supplied prices. A smile fit is a model
of total variance, and passing grid diagnostics is not a proof of global
arbitrage freedom.""",
    [
        (
            "Recover a supplied smile from option prices",
            """The raw SVI curve supplies synthetic total variances at a one-year expiry.
Black inversion should recover their square roots. Normal volatility uses
different units and also permits negative forwards and strikes.""",
            """
from convexity.math import implied_volatility as iv
smile=iv.SVI(.02,.07,-.3,.01,.2)
k=np.linspace(-.6,.6,41);forward=100*np.exp(.03);strikes=forward*np.exp(k)
vol=np.sqrt(smile(k))
premiums=[pr.forward_option(forward,s,v,1.,discount=np.exp(-.04)) for s,v in zip(strikes,vol)]
recovered=[iv.implied_volatility(p,forward,s,1.,discount=np.exp(-.04)) for p,s in zip(premiums,strikes)]
plt.plot(k,vol,label='Supplied');plt.scatter(k,recovered,s=12,label='Inverted');plt.xlabel('Log strike / forward');plt.ylabel('Annual volatility');plt.legend();plt.show()
print('Maximum inversion error:',np.max(np.abs(vol-recovered)))
""",
        ),
        (
            "Fit and inspect total variance",
            """The fitter imposes positivity, wing bounds and sampled butterfly constraints.
The surface report includes every supplied expiry plus intermediate times, so a
narrow calendar-arbitrage interval cannot disappear between requested samples.""",
            """
fitted=iv.SVI.fit(k,smile(k))
surface=iv.SmileSurface([1.,2.],(fitted,iv.SVI(.04,.09,-.3,.01,.2)))
times=np.linspace(1,2,35)
variance=np.array([surface.total_variance(t,k) for t in times])
plt.imshow(variance,origin='lower',aspect='auto',extent=[k.min(),k.max(),1,2]);plt.colorbar(label='Total variance');plt.xlabel('Log moneyness');plt.ylabel('Expiry years');plt.show()
low,high=iv.SVI(.01,0,0,0,.1),iv.SVI(.05,0,0,0,.1)
narrow=iv.SmileSurface([1.,1.001,1.002,2.],(low,high,low,high))
display(pd.DataFrame({'Surface':['Smooth increasing','Narrow calendar violation'],
                     'Grid report passed':[surface.diagnose(k).passed,narrow.diagnose(k,time_points=3).passed]}))
""",
        ),
        (
            "Density and local volatility are different derivatives",
            """The risk-neutral density is strike curvature of discounted call prices.
Dupire local volatility also depends on maturity derivatives and a consistent
surface. The constant-volatility example has an independent analytic answer.""",
            """
from scipy.stats import norm
strikes=np.linspace(55,160,100)
call=lambda strike:pr.forward_option(forward,strike,.2,1,discount=np.exp(-.04))
density=iv.risk_neutral_density(call,strikes,np.exp(-.04))
exact=norm.pdf((np.log(strikes/forward)+.5*.2**2)/.2)/(.2*strikes)
plt.plot(strikes,exact,label='Lognormal reference');plt.plot(strikes,density,'--',label='Call-price curvature');plt.xlabel('Terminal spot');plt.ylabel('Density');plt.legend();plt.show()
native=iv.black_variance_surface(ref,[expiry,'2028-01-01'],[50,100,150],np.full((3,2),.2))
display(pd.DataFrame([(t,s,iv.dupire_local_volatility(native,discount,dividend,100,t,s)) for t in [.5,1,1.5] for s in [80,100,120]],columns=['Time','Strike','Dupire local vol']))
""",
        ),
    ],
)

BOOKS["calibration"] = notebook(
    "Calibration: recovery, weighting and identifiability",
    2100,
    """Family 21. Quote uncertainty is generated explicitly. Calibration success,
parameter identifiability and predictive performance are different questions.
Local covariance estimates are residual-scaled linear approximations.""",
    [
        (
            "Recover volatility with heteroskedastic quote noise",
            """The precision weights match the generated observation noise. The reported
covariance estimates residual scale; it is not a Bayesian posterior.""",
            """
from convexity.math import calibration as ca
strikes=np.linspace(75,125,21)
def predict(p):
    return np.array([pr.forward_option(100.,s,p[0],1.,discount=np.exp(-.04)) for s in strikes])
truth=predict([.24]);noise=.015+.0015*np.abs(strikes-100)
observed=truth+rng.normal(size=len(strikes))*noise
fit=ca.calibrate(ca.CalibrationProblem(predict,observed,[.4],weights=1/noise**2,lower=[.01],upper=[1.]))
plt.errorbar(strikes,observed,yerr=2*noise,fmt='.',label='Observed ±2 supplied noise SD')
plt.plot(strikes,predict(fit.parameters),label='Fitted');plt.xlabel('Strike');plt.ylabel('Price');plt.legend();plt.show()
display(pd.DataFrame({'Truth':[.24],'Estimate':fit.parameters,'Local standard error':np.sqrt(np.diag(fit.covariance)),'RMSE':[fit.rmse]}))
""",
        ),
        (
            "A flat direction can remain after convergence",
            """This deliberately redundant parameterization identifies only the sum of two
parameters. A small residual does not identify either component. Singular values
and absent covariance reveal the problem.""",
            """
x=np.linspace(1,3,15)
redundant=lambda p:(p[0]+p[1])*x
result=ca.calibrate(ca.CalibrationProblem(redundant,3*x,[1.,1.]))
a=np.linspace(0,3,100);b=np.linspace(0,3,100);A,B=np.meshgrid(a,b)
loss=np.sum(x*x)*(A+B-3)**2
plt.contourf(A,B,np.log1p(loss),levels=20);plt.colorbar(label='log(1 + squared residual)')
plt.plot(a,3-a,color='white',label='Equivalent fits');plt.xlabel('Parameter 1');plt.ylabel('Parameter 2');plt.legend();plt.show()
display(pd.DataFrame({'parameter':result.parameters}));print('Rank:',result.rank,'Singular values:',result.singular_values,'Covariance:',result.covariance)
""",
        ),
        (
            "Bounds and regularization change the inverse problem",
            """A bound can prevent recovery despite a converged optimizer. Regularization
selects a solution relative to a supplied prior. Neither case receives an
unconstrained residual-based covariance.""",
            """
bounded=ca.calibrate(ca.CalibrationProblem(predict,truth,[.15],lower=[.01],upper=[.2]))
penalized=ca.calibrate(ca.CalibrationProblem(redundant,3*x,[1.,1.],prior=[0.,0.],regularization=10.))
display(pd.DataFrame({'fit':['Unrestricted volatility','Volatility capped at .20','Regularized redundant model'],
 'parameters':[fit.parameters.tolist(),bounded.parameters.tolist(),penalized.parameters.tolist()],
 'RMSE':[fit.rmse,bounded.rmse,penalized.rmse],
 'covariance available':[fit.covariance is not None,bounded.covariance is not None,penalized.covariance is not None]}))
plt.plot(strikes,predict(bounded.parameters)-truth,marker='.');plt.axhline(0,color='black',lw=.8);plt.xlabel('Strike');plt.ylabel('Bound-constrained pricing residual');plt.show()
""",
        ),
    ],
)

BOOKS["stopping"] = notebook(
    "Optimal stopping: envelopes and independent exercise policies",
    2200,
    """Family 22. Exercise opportunities and the information available at each time
are explicit. The finite-state problem has an exact recursion; regression-based
exercise is approximate and must be evaluated on independent paths.""",
    [
        (
            "Inspect an exact Snell envelope",
            """Rows are decision times and columns are states. The exercise region compares
immediate payoff with discounted continuation. Terminal exercise is mandatory.""",
            """
from convexity.math import stopping as st
rewards=np.array([[1.,1.],[3.,0.],[0.,6.]])
transitions=np.tile([[.5,.5],[.5,.5]],(2,1,1))
solution=st.BackwardInduction().solve(st.StoppingProblem(rewards,transitions,.9))
fig,axes=plt.subplots(1,3,figsize=(11,3))
for ax,values,title in zip(axes,[solution.values,solution.continuation,solution.exercise_region],['Value','Continuation','Exercise']):
    ax.imshow(values,aspect='auto');ax.set(title=title,xlabel='State',ylabel='Time index')
    for i in range(3):
        for j in range(2):ax.text(j,i,f'{float(values[i,j]):.3g}',ha='center',va='center',color='white')
plt.tight_layout();plt.show()
""",
        ),
        (
            "Train and evaluate on separate Q paths",
            """The state is sufficient for this vanilla put. Validation error is conditional
on the learned policy. The European payoff is a useful lower reference; a
binomial American price provides an independent numerical comparison.""",
            """
def paths(seed,draws):
    gen=np.random.default_rng(seed);dt=1/30
    logs=np.c_[np.zeros(draws),np.cumsum((.04-.5*.2**2)*dt+.2*np.sqrt(dt)*gen.normal(size=(draws,30)),axis=1)]
    spot=100*np.exp(logs)
    return spot,np.maximum(100-spot,0)
train,train_reward=paths(2201,14000)
validation,reward=paths(2202,22000)
policy=st.least_squares_policy(train,train_reward,np.exp(-.04/30),degree=3)
result=policy.evaluate(validation,reward,np.exp(-.04/30))
stock_model=pr.BlackScholesModel(100,discount,cv.flat_curve(ref,0),.2)
contract=ct.VanillaContract(100,ct.Exercise('american',(ref,expiry)),'put')
reference=pr.price(pr.PricingProblem(contract,stock_model,ref),pr.VanillaEngine('tree',steps=600))
display(pd.DataFrame({'Method':['European MC','Independent policy MC','American tree'],
 'Value':[reward[:,-1].mean()*np.exp(-.04),result.value,reference.value],
 'Policy standard error':[None,result.standard_error,None]}))
tau=result.exercise_indices;early=tau<30
plt.scatter(tau[early]/30,validation[np.arange(len(tau))[early],tau[early]],s=3,alpha=.07)
plt.xlabel('Exercise time');plt.ylabel('Spot at early exercise');plt.title('Exercise decisions on validation paths');plt.show()
""",
        ),
        (
            "Exercise dates and dividend events",
            """A Bermudan contract has a finite exercise set. Discrete cash dividends are
passed to the finite-difference engine as dated events, rather than smoothed into
a yield.""",
            """
rows=[]
for style,dates in [('european',(expiry,)),('bermudan',('2026-07-01',expiry)),('american',(ref,expiry))]:
    contract=ct.VanillaContract(100,ct.Exercise(style,dates),'call')
    problem=pr.PricingProblem(contract,stock_model,ref)
    for amount in [0.,4.]:
        divs=() if amount==0 else (ct.CashFlow('2026-07-01',amount),)
        rows.append((style,amount,pr.price(problem,pr.VanillaEngine('fd',steps=200,grid=200,dividends=divs)).value))
display(pd.DataFrame(rows,columns=['Exercise','Cash dividend','Call value']))
""",
        ),
    ],
)

BOOKS["sensitivities"] = notebook(
    "Sensitivities and hedging geometry",
    2300,
    """Family 23. All derivatives specify which input is bumped and its unit. A
hedge minimizes residual exposure in supplied risk coordinates; it is not an
optimal trading strategy or a promise of stable realized P&L.""",
    [
        (
            "Analytic delta and finite-difference step choice",
            """Central differences trade truncation error against cancellation. The test
holds dividend yield, financing and implied volatility fixed when spot changes.""",
            """
from dataclasses import replace
from convexity.math import sensitivities as se
contract=ct.VanillaContract(100,ct.Exercise('european',(expiry,)))
def value(spot):
    return pr.price(pr.PricingProblem(contract,replace(model,spot=spot),ref),pr.VanillaEngine()).value
analytic=pr.price(pr.PricingProblem(contract,model,ref),pr.VanillaEngine())
steps=np.logspace(-6,0,25)
estimates=np.array([(value(100+h)-value(100-h))/(2*h) for h in steps])
plt.loglog(steps,np.maximum(np.abs(estimates-analytic.greeks['delta']),1e-16),marker='.')
plt.xlabel('Absolute spot bump');plt.ylabel('Absolute delta error');plt.title('Step size matters');plt.show()
display(pd.DataFrame([dict(analytic.greeks)]))
""",
        ),
        (
            "Cross-Greeks have two input units",
            """The Hessian below uses spot units and decimal volatility units. Its mixed
entry is the sensitivity of delta to volatility; comparing raw entries across
coordinates requires attention to their scales.""",
            """
def price_at(x):
    changed=replace(model,spot=x[0],volatility=x[1])
    return pr.price(pr.PricingProblem(contract,changed,ref),pr.VanillaEngine()).value
risk=se.bump_sensitivities(price_at,[100.,.2],steps=[.01,1e-4])
display(pd.DataFrame(risk['hessian'],index=['Spot','Decimal vol'],columns=['Spot','Decimal vol']))
plt.imshow(risk['hessian']);plt.xticks([0,1],['Spot','Vol']);plt.yticks([0,1],['Spot','Vol']);plt.colorbar(label='Second derivative');plt.show()
""",
        ),
        (
            "Exposure matrices, bounds and nullspaces",
            """Rows of A are risk factors and columns are hedge instruments. The target is an
existing portfolio exposure, so A h + target is the residual. Nonnegative
holdings restrict the feasible risk geometry.""",
            """
A=np.array([[1.,0.,1.],[0.,1.,1.]])
target=np.array([2.,-1.])
free=se.hedge(A,target)
bounded=se.hedge(A,target,lower=0)
budgeted=se.hedge(A,target,equality=[[1.,1.,1.]],equality_target=[0.])
display(pd.DataFrame([r.holdings for r in [free,bounded,budgeted]],index=['Unconstrained','Long only','Zero net units'],columns=['Instrument 1','Instrument 2','Instrument 3']))
pd.DataFrame([r.residual_exposure for r in [free,bounded,budgeted]],index=['Unconstrained','Long only','Zero net units'],columns=['Risk 1','Risk 2']).plot.bar()
plt.ylabel('Residual exposure');plt.tight_layout();plt.show()
print('Exposure nullspace basis:',free.nullspace.ravel())
""",
        ),
    ],
)

BOOKS["fixed_income"] = notebook(
    "Bonds: yield, accrual, optionality and curve risk",
    2400,
    """Family 24. Clean and dirty prices are per 100 outstanding principal. Yield
duration describes promised cash flows; effective callable-bond risk requires
repricing the exercise option under each perturbed curve.""",
    [
        (
            "Price/yield curvature and local duration",
            """A duration-only approximation is local. Adding convexity improves a small
parallel yield change but does not capture every curve or optionality change.""",
            """
from convexity.math import fixed_income as fi, rates as ra
schedule=ct.schedule('2025-01-01','2031-01-01')
bond=fi.fixed_bond(schedule,.05)
base=fi.bond_analytics(bond,discount)
yields=np.linspace(.005,.09,60)
exact=np.array([fi.price_from_yield(bond,y,ref) for y in yields])
dy=yields-base.yield_to_maturity
linear=base.clean-base.dirty*base.duration*dy
quadratic=linear+.5*base.dirty*base.convexity*dy**2
plt.plot(yields,exact,label='Repriced');plt.plot(yields,linear,'--',label='Duration');plt.plot(yields,quadratic,':',label='Duration + convexity')
plt.xlabel('Annual yield');plt.ylabel('Clean price');plt.legend();plt.show()
display(pd.DataFrame([{'Clean':base.clean,'Dirty':base.dirty,'Accrued':base.accrued,'Yield':base.yield_to_maturity,'Duration':base.duration,'Convexity':base.convexity,'DV01':base.dv01}]))
""",
        ),
        (
            "Accrual and amortization",
            """Coupon dates reset accrued interest. An amortizing bond reduces its outstanding
principal, which changes both cash flows and the meaning of a per-100 quote.""",
            """
dates=pd.date_range('2026-01-02','2026-12-30',freq='7D')
accrual=[fi.bond_analytics(bond,discount,settlement_date=d.date(),compute_yield=False).accrued for d in dates]
plt.plot(dates,accrual);plt.ylabel('Accrued per 100');plt.title('Accrual between semiannual coupons');plt.xticks(rotation=25);plt.tight_layout();plt.show()
amort=fi.amortizing_bond(ct.schedule(ref,'2029-01-01',months=12),[100,70,40],.05)
display(pd.DataFrame([(ct.to_date(cf.date).isoformat(),cf.amount) for cf in fi.cashflows(amort,ref)],columns=['Date','Amortizing cash flow']))
""",
        ),
        (
            "Callable bonds must use an exercise engine",
            """The issuer's call option reduces bond value. For key-rate sensitivity, the
engine factory rebuilds the curve-fitted rate model for every bumped curve.""",
            """
callable_bond=fi.callable_bond(schedule,.05,[('2028-01-01',100,'call')])
def engine(curve):
    return fi.callable_bond_engine(curve,mean_reversion=.1,volatility=.01,steps=160)
rows=[]
for rate in np.linspace(.015,.07,12):
    curve=cv.flat_curve(ref,rate)
    rows.append((rate,fi.bond_analytics(bond,curve,compute_yield=False).dirty,
                 fi.bond_analytics(callable_bond,curve,engine=engine(curve),compute_yield=False).dirty))
prices=pd.DataFrame(rows,columns=['Rate','Straight','Callable'])
prices.plot(x='Rate',y=['Straight','Callable']);plt.ylabel('Dirty price');plt.show()
keys=[ref,'2028-01-01','2031-01-01']
display(pd.DataFrame({'Key date':keys,'Straight KRD':fi.key_rate_duration(bond,discount,keys),
 'Callable effective KRD':fi.key_rate_duration(callable_bond,discount,keys,engine_factory=engine)}))
""",
        ),
        (
            "Carry and roll-down assumptions",
            """Holding the dated forward curve and holding the zero curve fixed by remaining
tenor are different scenarios. The implied-forward roll has zero excess return
by construction, not because it predicts tomorrow's curve.""",
            """
curve=cv.zero_curve([ref,'2027-01-01','2029-01-01','2031-01-01'],[.02,.03,.05,.06])
zero=fi.zero_coupon('2031-01-01')
rows=[]
for evolution in ['constant_tenor','implied_forward']:
    r=fi.carry_roll_down(zero,curve,'2027-01-01',evolution=evolution)
    rows.append({'Evolution':evolution,**dict(r.items())})
display(pd.DataFrame(rows))
""",
        ),
    ],
)

BOOKS["rates"] = notebook(
    "Rates and inflation: dynamics, numeraires and options",
    2500,
    """Family 25. P forecasting, bank-account Q valuation and terminal-forward
simulation are labeled separately. Inflation dynamics below use deterministic
nominal and real curves; no stochastic rate/CPI correlation is inferred.""",
    [
        (
            "Curve-fitted and standalone short-rate models",
            """Hull–White and G2 fit the supplied discount curve at time zero. Vasicek and CIR
use supplied parameters and need not reproduce that curve at every maturity.""",
            """
from convexity.math import rates as ra
models={
 'Vasicek':ra.ShortRateModel('vasicek',dict(r0=.04,kappa=.2,theta=.05,sigma=.01)),
 'CIR':ra.ShortRateModel('cir',dict(r0=.04,kappa=.2,theta=.05,sigma=.1)),
 'Hull-White':ra.ShortRateModel('hull_white',dict(kappa=.1,sigma=.01),discount),
 'G2':ra.ShortRateModel('g2',dict(a=.1,sigma=.01,b=.2,eta=.015,rho=-.5),discount)}
t=np.linspace(.1,10,70)
for name,m in models.items():
    state=[0.,0.] if name=='G2' else .04
    plt.plot(t,[-np.log(m.discount_bond(0,T,state))/T for T in t],label=name)
plt.xlabel('Maturity years');plt.ylabel('Time-zero continuous zero rate');plt.legend();plt.show()
physical=ra.ShortRateModel('vasicek',dict(r0=.04,kappa=.4,theta=.06,sigma=.015),measure='P')
paths=physical.process().sample(.04,np.linspace(0,5,60),draws=12,rng=rng)
plt.plot(paths.times,paths.values.T,alpha=.5);plt.xlabel('Years');plt.ylabel('Short rate');plt.title('Explicit P forecast paths');plt.show()
""",
        ),
        (
            "Native HJM and a terminal-forward market model",
            """QuantLib GSR provides a Gaussian HJM model. The market-model simulation uses a
terminal bond numeraire: the last forward rate has a martingale reference under
that measure, unlike an arbitrary bank-account-measure forward rate.""",
            """
gsr=ra.gaussian_hjm(discount,['2027-01-01'],[.01,.015],[.1])
print('GSR initial 5-year discount:',ra.gaussian_bond_price(gsr,5.))
rate_times=[.5,1.,1.5,2.]
market_model=ra.forward_market_model(rate_times,[.04,.04,.04],volatility=.2,factors=2)
paths=ra.market_model_paths(market_model,draws=2500,seed=2517)
last=paths.values[:,:,-1]
plt.plot(paths.times,last[:30].T,alpha=.15,color='steelblue')
plt.plot(paths.times,last.mean(axis=0),color='black',label='Last forward mean');plt.axhline(.04,ls='--',label='Initial forward')
plt.xlabel('Evolution time');plt.ylabel('Last forward');plt.legend();plt.show()
print('Simulation measure:',paths.measure)
""",
        ),
        (
            "Inflation levels and Q valuation",
            """P expected CPI growth is an empirical assumption. With deterministic rates,
the Q forward CPI uses D_real / D_nominal. The CPI option is an option on the
index level; native inflation contracts additionally specify publication lags.""",
            """
physical_cpi=ra.InflationDynamics(.1,'P',drift=.035)
pricing_cpi=ra.InflationDynamics(.1,'Q',nominal=discount,real=cv.flat_curve(ref,.02))
t=np.linspace(0,5,60)
plt.plot(t,[physical_cpi.transition(100,0,T).mean() for T in t],label='P expected CPI')
plt.plot(t,[pricing_cpi.transition(100,0,T).mean() for T in t],label='Q expected CPI');plt.xlabel('Years');plt.ylabel('CPI');plt.legend();plt.show()
display(pd.DataFrame([(s,pricing_cpi.option(100,s,2)) for s in [90,100,110,120]],columns=['CPI strike','Q two-year call']))
""",
        ),
        (
            "Caps under different volatility specifications",
            """A normal quote and a Black quote have different units. The model-based rate
option engine uses the same discount curve that fitted the Hull–White model.""",
            """
dates=ct.schedule('2027-01-01','2030-01-01')
cap=ra.cap_floor(dates,ra.rate_index(discount),.04)
rows=[]
for method,vol in [('black',.2),('normal',.01),('analytic',None),('tree',None)]:
    result=ra.rate_option_price(cap,discount,method=method,volatility=vol,model=models['Hull-White'])
    rows.append((method,vol,result.value))
display(pd.DataFrame(rows,columns=['Engine','Volatility input','Cap value']))
""",
        ),
    ],
)

BOOKS["credit"] = notebook(
    "Credit: survival, recovery, migration and structural default",
    2600,
    """Family 26. Reduced-form default timing and Merton terminal insolvency are
separate models. P default estimation is kept distinct from Q pricing; all
hazards, recoveries and asset values are synthetic.""",
    [
        (
            "Survival and censored default-time simulation",
            """A draw of infinity means survival beyond the requested simulation horizon.
It does not mean the obligor can never default.""",
            """
from convexity.math import credit as cr
hazard=cr.flat_hazard(ref,.06)
t=np.linspace(0,10,80)
plt.plot(t,[hazard.survival(T) for T in t],label='Survival')
plt.plot(t,[hazard.default_probability(0,T) for T in t],label='Cumulative default');plt.xlabel('Years');plt.ylabel('Probability');plt.legend();plt.show()
draws=hazard.sample(5000,horizon=5,rng=rng)
display(pd.DataFrame({'quantity':['Default by 5, sampled','Default by 5, analytic','Censored draws'],
 'value':[np.isfinite(draws).mean(),1-np.exp(-.06*5),np.isinf(draws).sum()]}))
""",
        ),
        (
            "Recovery of par and CDS spread exposure",
            """Risky cash flows include recovery paid at default. The CDS table includes a
current trade date and accrued-premium rebate, so net premium exposure differs
from the coupon leg alone.""",
            """
recoveries=np.linspace(0,.8,30)
values=[cr.risky_cashflows([5.],[100.],discount,hazard,recovery=r,recovery_notional=100) for r in recoveries]
plt.plot(recoveries,values);plt.xlabel('Recovery fraction');plt.ylabel('Risky 5-year zero value');plt.show()
dates=ct.schedule('2025-12-20','2030-12-20',months=3)
kwargs=dict(trade_date=ref)
cds=cr.cds_price(dates,.03,discount,hazard,**kwargs)
par=cr.cds_price(dates,cds.fair_spread,discount,hazard,**kwargs)
display(pd.DataFrame([dict(cds.items())]));print('Repriced par CDS NPV:',par.value)
""",
        ),
        (
            "Rating migration is a full transition law",
            """The final state is absorbing default. The continuous-time generator is
estimated from transition counts and exposure time, not by taking an arbitrary
matrix logarithm of a discrete transition matrix.""",
            """
migration=cr.MigrationModel([[-.10,.08,.02],[.02,-.22,.20],[0.,0.,0.]])
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for ax,time in zip(axes,[1,5]):
    matrix=migration.transition(time).transition
    ax.imshow(matrix,vmin=0,vmax=1);ax.set(title=f'{time}-year transition',xlabel='To state',ylabel='From state')
    for i in range(3):
        for j in range(3):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',color='white')
plt.show()
print('P censored hazard estimate:',cr.estimate_hazard([2.,3.,5.],[True,False,True]))
""",
        ),
        (
            "Structural terminal default and the capital structure",
            """Merton values equity as an option on assets and debt as the residual promised
claim with default loss. The direct debt formula avoids numerical cancellation
when asset value is very large. P and Q default probabilities use distinct drifts.""",
            """
assets=np.linspace(60,180,100)
results=[cr.merton_structural(v,100,.25,2,.04,physical_drift=.09) for v in assets]
fig,axes=plt.subplots(1,2,figsize=(11,3.5))
axes[0].plot(assets,[r.equity for r in results],label='Equity');axes[0].plot(assets,[r.debt for r in results],label='Debt');axes[0].set(xlabel='Asset value',ylabel='Claim value');axes[0].legend()
axes[1].plot(assets,[r.p_default_probability for r in results],label='P');axes[1].plot(assets,[r.q_default_probability for r in results],label='Q');axes[1].set(xlabel='Asset value',ylabel='Terminal default probability');axes[1].legend();plt.show()
""",
        ),
    ],
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only", nargs="*", choices=list(BOOKS))
    args = parser.parse_args()
    for name in args.only or BOOKS:
        path = ROOT / "notebooks" / f"{name}.ipynb"
        nb = BOOKS[name]
        if args.execute:
            from nbclient import NotebookClient
            from jupyter_client.manager import KernelManager

            km = KernelManager(kernel_name="python3")
            spec = km.kernel_spec
            if spec is None:
                raise RuntimeError("python3 kernel specification is unavailable")
            spec.argv = [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ]
            client = NotebookClient(
                nb, timeout=300, km=km, resources={"metadata": {"path": str(ROOT)}}
            )
            try:
                client.execute()
            finally:
                if km.has_kernel:
                    km.shutdown_kernel(now=True)
        nbf.validate(nb)
        nbf.write(nb, path)
        print(
            f"{name}: {len(nb.cells)} cells"
            + (" executed" if args.execute else " built"),
            flush=True,
        )


if __name__ == "__main__":
    main()
