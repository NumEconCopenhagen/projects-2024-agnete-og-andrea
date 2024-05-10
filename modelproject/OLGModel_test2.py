from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import sympy as sm
'''
def OLG_analytical(print_output=True):
    ##solving model for k*
    ##we define the parameters and functions:
    from IPython.display import display, Math

    # Define parameters and variables
    beta = sm.symbols('beta')
    alpha = sm.symbols('alpha')
    theta = sm.symbols('theta')
    sigma = sm.symbols('sigma')
    n = sm.symbols('n')
    s = sm.symbols('s')
    rt = sm.symbols('r_t')
    wt = sm.symbols('w_t')
    Kt = sm.symbols('K_t')
    Kt_1 = sm.symbols('K_t-1')
    Lt = sm.symbols('L_t')
    kt = sm.symbols('k_t')
    kt_1 = sm.symbols('k_{t-1}')
    k = sm.symbols('k*')
    # Define production function
    f = (alpha*Kt_1**(-theta)+(1-alpha)*(Lt)**(-theta))**(-1.0/theta)

    # Solve for wage and MPK
    mpl = sm.diff(f, Lt)

    #substitute in kt
    mpl_sub = mpl.subs(alpha * Kt**(-theta) + (1 - alpha) * Lt**(-theta)**(-1.0/theta), kt**alpha)
    w_eq = sm.Eq(wt, mpl_sub)

    # Define household utility
    C1t, C2t_1 = sm.symbols('C_1t C_{2t+1}')
    wt, rt_1 = sm.symbols('w_t, r_{t+1}')
    St = sm.symbols('S_t')
    C1t = (1-s)*wt
    C2t_1 = (1+rt_1)*Kt
    st = 1-C1t
    U = C1t**(1-sigma)/(1-sigma)+beta*C2t_1**(1-sigma)/(1-sigma)

    # Find the derivative of U with respect to st
    dU = sm.diff(U, 1-C1t)

    # Solve for s_t
    s_eq = sm.Eq(0, dU)
    st_path = sm.solve(s_eq, s)[0]

    # Define the equation for capital accumulation (kt_1)
    kt_1_eq = sm.Eq(kt_1, st_path * wt / (1 + n))

    # Solve for steady state capital (k*)
    ss_solve = sm.Eq(kt, kt_1)
    k_ss = sm.solve(ss_solve, kt)[0]

    # Analytical solution with chosen parameter values
    f_kss = sm.lambdify((alpha, theta, sigma, n, beta, wt, rt_1, Kt_1), k_ss)

    if print_output:
        # Print the resulting equations
        print("Wage equation for w_t:")
        display(Math(sm.latex(w_eq)))

        print("Derivative of utility with respect to savings (s):")
        display(Math(sm.latex(dU)))

        print("Optimal savings rate (s):")
        display(Math(sm.latex(st_path)))

        print("Equation for capital accumulation:")
        display(Math(sm.latex(kt_1_eq)))

        print("Steady state for capital per capita:")
        display(Math(sm.latex(k_ss)))

    return f_kss
'''

class OLGModelClass():

    def __init__(self,do_print=True):
        """ create the model """

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()

    def setup(self):
        """ baseline parameters """

        par = self.par

        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.rho = 0.10 # discount factor
        par.d = 1.0 # contributions to old
        par.n = 0.02 # population growth rate

        # b. firms
        par.production_function = 'cobb-douglas'
        par.alpha = 0.30 # capital weight
        par.delta = 1.0 # depreciation rate

        # c. misc
        par.K_ini = 0.1 # initial capital stock
        par.L_ini = 1.0 # initial labor stock
        par.simT = 50 # length of simulation

    def allocate(self):
        """ allocate arrays for simulation """
        
        par = self.par
        sim = self.sim

        # a. list of variables
        household = ['C1','C2']
        firm = ['K','L','Y']
        prices = ['w','r']

        # b. allocate
        allvarnames = household + firm + prices 
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)

    def simulate(self,do_print=True):
        """ simulate model """

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values
        sim.K[0] = par.K_ini
        sim.L[0] = par.L_ini

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before s
            simulate_before_s(par,sim,t)

            if t == par.simT-1: continue          

            # i. find bracket to search
            s_min,s_max = find_s_bracket(par,sim,t)

            # ii. find optimal s
            obj = lambda s: calc_euler_error(s,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=(s_min,s_max),method='bisect')
            s = result.root

            # iii. simulate after s
            simulate_after_s(par,sim,t,s)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def find_s_bracket(par,sim,t,maxiter=500,do_print=False):
    """ find bracket for s to search in """

    # a. maximum bracket
    s_min = 0.0 + 1e-8 # save almost nothing
    s_max = 1.0 - 1e-8 # save almost everything

    # b. saving a lot is always possible 
    value = calc_euler_error(s_max,par,sim,t)
    sign_max = np.sign(value)
    if do_print: print(f'euler-error for s = {s_max:12.8f} = {value:12.8f}')

    # c. find bracket      
    lower = s_min
    upper = s_max

    it = 0
    while it < maxiter:
                
        # i. midpoint and value
        s = (lower+upper)/2 # midpoint
        value = calc_euler_error(s,par,sim,t)

        if do_print: print(f'euler-error for s = {s:12.8f} = {value:12.8f}')

        # ii. check conditions
        valid = not np.isnan(value)
        correct_sign = np.sign(value)*sign_max < 0
        
        # iii. next step
        if valid and correct_sign: # found!
            s_min = s
            s_max = upper
            if do_print: 
                print(f'bracket to search in with opposite signed errors:')
                print(f'[{s_min:12.8f}-{s_max:12.8f}]')
            return s_min,s_max
        elif not valid: # too low s -> increase lower bound
            lower = s
        else: # too high s -> increase upper bound
            upper = s

        # iv. increment
        it += 1

    raise Exception('cannot find bracket for s')

def calc_euler_error(s,par,sim,t):
    """ target function for finding s with bisection """

    # a. simulate forward
    simulate_after_s(par,sim,t,s)
    simulate_before_s(par,sim,t+1) # next period

    # c. Euler equation
    LHS = 1/sim.C1[0]
    RHS = ((1+sim.r[1])/(1+par.rho))*(1/sim.C2[1])

    return LHS-RHS

def simulate_before_s(par,sim,t):
    """ simulate forward """

    if t == 0:
        sim.K[t] = par.K_ini
        sim.L[t] = par.L_ini

    if t > 0:
        sim.L[t] = sim.L[t-1]*(1+par.n)

    # a. production and factor prices
    if par.production_function == 'ces':

        # i. production
        sim.Y[t] = ( par.alpha*sim.K_lag[t]**(-par.theta) + (1-par.alpha)*(1.0)**(-par.theta) )**(-1.0/par.theta)

        # ii. factor prices
        sim.rk[t] = par.alpha*sim.K_lag[t]**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)
        sim.w[t] = (1-par.alpha)*(1.0)**(-par.theta-1) * sim.Y[t]**(1.0+par.theta)

    elif par.production_function == 'cobb-douglas':

        # i. production
        sim.Y[t] = sim.K[t]**par.alpha * (sim.L[t])**(1-par.alpha)

        # ii. factor prices
        sim.r[t] = par.alpha * sim.K[t]**(par.alpha-1) * (sim.L[t])**(1-par.alpha)
        sim.w[t] = (1-par.alpha) * sim.K[t]**(par.alpha) * (sim.L[t])**(-par.alpha)

    else:

        raise NotImplementedError('unknown type of production function')

    # c. consumption
    sim.C2[t] = (1+par.n)*(1+sim.r[t])*(sim.K[t])+(1+par.n)*par.d

def simulate_after_s(par,sim,t,s):
    """ simulate forward """

    # a. consumption of young
    sim.C1[t] = sim.w[t]-(1+par.n)*sim.K[t+1]+par.d

    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] 
    sim.K[t+1] = (1-par.delta)*sim.K[t]+I/(1+par.n)