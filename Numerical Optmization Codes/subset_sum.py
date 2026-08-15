import collections
import random
from random import uniform
from math import *
import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize
import concurrent.futures
import warnings

# ============================================================================
# Provide Inputs
# ============================================================================

# Number of parallel processes
Num_cores = 20

# Number of random optimization runs
TOTAL_RUNS = 100



# ============================================================================
# Basic Functions
# ============================================================================

def H(c):
    """
    Entropy function
    """
    if c == 0. or c == 1.:
        return 0.
    
    if c < 0. or c > 1.:
        return -1000
    
    return -(c * log2(c) + (1 - c) * log2(1 - c))

def binomH(n, k):
    """
    Asymptotic exponent of binomial coefficient.
    """
    if n <= 0:
        return 0.0

    return n * H(k / n)


def multiH(n, c):
    """
    Asymptotic exponent of multinomial coefficient.
    """
    if sum(c)>n:
        return 0
    tot=0
    val=n
    for i in c:
        tot+=binomH(n,i)
        n-=i
    return tot


def wrap(f,g) :
    def inner(x):
        return f(g(*x))
    return inner

def r(x,y,z):
    return [(ru(x,y)) for i in range(z)]



# ============================================================================
# Variable Wrapper
# ============================================================================

set_vars = collections.namedtuple('Subset_Sum', 'b g lmd')

num_params = 3

def subsum(f) : return wrap(f, set_vars)

# ============================================================================
# Domain Size and the counts of representations exponents
# ============================================================================

Tao = lambda x : binomH((1-x.g)/4, (1-x.g*x.b)/8) + binomH(x.g, x.g*x.b/8)
S = lambda x : 4*binomH((1-x.g)/4, (1-x.g*x.b)/8) + binomH(x.g, x.g*x.b/2)

R1 = lambda x : x.g*x.b/2
R2 = lambda x : x.g*x.b/4


# ============================================================================
# Objective Function
# ============================================================================

def time(x):
    
    x = set_vars(*x)

    good_D = 1 - S(x)

    good_R = max(0, Tao(x) - R1(x))

    collisions_when_R_good = max(0, Tao(x) - 2*R2(x))

    one_collision = Tao(x)

    return good_R + collisions_when_R_good + one_collision + good_D

# ============================================================================
# Constraints
# ============================================================================


constraints = [

    { 'type' : 'eq',     'fun' : subsum(lambda x : x.lmd/2 - Tao(x))},

    
    { 'type' : 'ineq',   'fun' : subsum(lambda x : 1 - S(x))},
    
    { 'type' : 'ineq',   'fun' : subsum(lambda x : x.lmd - S(x))},
    
    { 'type' : 'ineq',   'fun' : subsum(lambda x : Tao(x) - R1(x))},
    
    { 'type' : 'ineq',   'fun' : subsum(lambda x : Tao(x) - 2*R2(x))},

    { 'type' : 'ineq',   'fun' : subsum(lambda x : (1-x.g)/4- (1-x.g*x.b)/8)},
    
    { 'type' : 'ineq',   'fun' : subsum(lambda x : (1-x.g*x.b))},

    { 'type' : 'ineq',   'fun' : subsum(lambda x : R1(x))},
    
    { 'type' : 'ineq',   'fun' : subsum(lambda x : R2(x))},

]

# ============================================================================
# Single Optimization Run
# ============================================================================
from random import uniform as ru
def single_optimization_run(run_id):
    """Worker function to perform a single multi-start optimization."""
    # Generate a random starting point for this specific run
    import warnings
    warnings.filterwarnings(
        "ignore",
        message="Values in x were outside bounds during a minimize step",
        category=RuntimeWarning,
        module="scipy.optimize"
    )
    
    start = [ru(0, 1) for _ in range(3)]

    bounds=[(0.,1)]*3
    
    result = minimize(time, start, 
                      bounds=bounds, tol=1e-9, 
                      constraints=constraints, options={'maxiter': 1000})
    
    return result.fun, result.x, result.success


# ============================================================================
# Main Optimization Loop
# ============================================================================

if __name__ == '__main__':
    lst=[]
    ress = 100
    best_params = None
    total_runs = TOTAL_RUNS
    
    print(f"Starting {total_runs} optimization runs across {Num_cores} cores...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=Num_cores) as executor:
        futures = [executor.submit(single_optimization_run, j) for j in range(total_runs)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                res, params, success = future.result()
                
                if success and res > 0 and res < ress:
                    ress = res
                    best_params = params
                    
            except Exception as e:
                print(f"A worker process encountered an error: {e}")

    if best_params is not None:
        print("\n=== Final Optimization Results ===")
        print(f"Optimized Function Value: Time exponent = {ress}" )
        print("Best Parameters:", best_params,"\n\n")
    else:
        print("\nNo successful optimization runs found that met the criteria (res > 0).")