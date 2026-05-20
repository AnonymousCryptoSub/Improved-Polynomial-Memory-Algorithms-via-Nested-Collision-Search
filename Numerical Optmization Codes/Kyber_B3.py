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
TOTAL_RUNS = 1000



# ============================================================================
# Weight distribution for Kyber B(3)
# ============================================================================

w1 = 15/64

w2 = 6/64

w3 = 1/64

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

set_vars = collections.namedtuple('LWE', ' t0_0 t1_0 th_0 z1_0 z2_0 z3_0 o1_0 o2_0 t0_1 t1_1 th_1 z1_1 z2_1 z3_1 o1_1 o2_1 g b1 b2 b3 lmd ')

num_params = 20

def lwe(f) : return wrap(f, set_vars)



# ============================================================================
# Level-0 Parameters
# ============================================================================

n3_0 = lambda x : x.b3 *x.g * w3
n2_0 = lambda x : x.b2 *x.g * w2
n1_0 = lambda x : x.b1 *x.g * w1
n0_0 = lambda x : x.g - 2*n3_0(x) - 2*n2_0(x) - 2*n1_0(x)


# ============================================================================
# Level-1 Parameters
# ============================================================================

n3_1 = lambda x : x.th_0 + x.t1_0 + x.o2_0 + x.z3_0
n2_1 = lambda x : n3_0(x)/2 - x.th_0 + x.o1_0 + x.z2_0 + x.t0_0 + x.o2_0           
n1_1 = lambda x : n3_0(x)/2 - x.th_0 + n2_0(x) -2*(x.t0_0+x.t1_0) + n1_0(x)/2 - x.o2_0  + x.z1_0 + x.t1_0           
n0_1 = lambda x : n0_0(x) + n1_0(x) - 2*(x.z1_0 + x.z2_0 + x.z3_0 + x.o1_0 + x.o2_0 - x.th_0 - x.t0_0)


### Layer-1 representation count exponent.
R1 =  lambda x : 2*multiH(n2_0(x),[x.t0_0, x.t1_0, x.t0_0, x.t1_0]) + 2*multiH(n1_0(x),[x.o1_0, x.o1_0, x.o2_0, x.o2_0, n1_0(x)/2 - x.o1_0 -x.o2_0]) + multiH(n0_0(x), [x.z1_0, x.z2_0, x.z3_0, x.z3_0, x.z1_0, x.z2_0]) + 2*multiH(n3_0(x),[x.th_0, x.th_0, n3_0(x)/2 - x.th_0])







# ============================================================================
# Level-2 Parameters
# ============================================================================

n3_2 = lambda x : x.th_1 + x.t1_1 + x.o2_1 + x.z3_1
n2_2 = lambda x : n3_1(x)/2 - x.th_1 + x.o1_1 + x.z2_1 + x.t0_1 + x.o2_1           
n1_2 = lambda x : n3_1(x)/2 - x.th_1 + n2_1(x) -2*(x.t0_1+x.t1_1) + n1_1(x)/2 - x.o2_1  + x.z1_1 + x.t1_1           
n0_2 = lambda x : n0_1(x) + n1_1(x) - 2*(x.z1_1 + x.z2_1 + x.z3_1 + x.o1_1 + x.o2_1 - x.th_1 - x.t0_1)



###  Layer-2 representation count exponent.
R2 =  lambda x : 2*multiH(n2_1(x),[x.t0_1, x.t1_1, x.t0_1, x.t1_1]) + 2*multiH(n1_1(x),[x.o1_1, x.o1_1, x.o2_1, x.o2_1, n1_1(x)/2 - x.o1_1 -x.o2_1]) + multiH(n0_1(x), [x.z1_1, x.z2_1, x.z3_1, x.z3_1, x.z1_1, x.z2_1]) + 2*multiH(n3_1(x),[x.th_1, x.th_1, n3_1(x)/2 - x.th_1])





# ============================================================================
# Domain Size Exponents
# ============================================================================


###  Exponent corresponding to the lower-level search domain.
domain = lambda x : multiH(x.g,[n3_2(x), n3_2(x), n2_2(x), n2_2(x), n1_2(x), n1_2(x)])+ multiH((1-x.g)/4,[(1-x.g*x.b1)*w1/4,(1-x.g*x.b2)*w2/4,(1-x.g*x.b1)*w1/4,(1-x.g*x.b2)*w2/4,(1-x.g*x.b3)*w3/4,(1-x.g*x.b3)*w3/4])



### Exponent corresponding to the restricted secret domain.
ss = lambda x : multiH(x.g,[n1_0(x) , n1_0(x), n2_0(x), n2_0(x), n3_0(x), n3_0(x)]) + 4*multiH((1-x.g)/4,[(1-x.g*x.b1)*w1/4,(1-x.g*x.b2)*w2/4,(1-x.g*x.b3)*w3/4,(1-x.g*x.b1)*w1/4,(1-x.g*x.b2)*w2/4,(1-x.g*x.b3)*w3/4])



# ============================================================================
# Objective Function
# ============================================================================

def time(x):

    x = set_vars(*x)

    good_D = multiH(1,[w1, w1, w2, w2, w3, w3]) - ss(x)

    good_R = max(0, domain(x) - R1(x))

    collisions_when_R_good = max(0, domain(x) - 2 * R2(x))

    one_collision = domain(x)

    return (
        good_R
        + collisions_when_R_good
        + one_collision
        + good_D
    )


# ============================================================================
# Constraints
# ============================================================================

constraints = [

    # Main balancing condition for Rho
    { 'type' : 'eq',     'fun' : lwe(lambda x : ss(x)*(1/2+x.lmd)-domain(x))},
    
    # Require enough middle-layer representations
    { 'type' : 'ineq',   'fun' : lwe(lambda x : domain(x) - R1(x))},
    
    # Require enough Layer-2 representations
    { 'type' : 'ineq',   'fun' : lwe(lambda x : domain(x) - 2*R2(x))},

    # Domain restriction
    { 'type' : 'ineq',   'fun' : lwe(lambda x :  multiH(1,[w1, w1, w2, w2, w3, w3]) - ss(x))},

    # Feasibility condition
    { 'type' : 'ineq',   'fun' : lwe(lambda x : (1-x.g)/4 - (1-x.g*x.b1)*w1/2 - (1-x.g*x.b2)*w2/2 - (1-x.g*x.b3)*w3/2)},
    

    # ========================================================================
    # Constraints in Level 0
    # ========================================================================
    { 'type' : 'ineq',   'fun' : lwe(lambda x : x.g-n0_0(x))},
    
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n0_0(x))},



    # ========================================================================
    # Constraints in Level 1
    # ========================================================================

    { 'type' : 'ineq',   'fun' : lwe(lambda x : n3_0(x)/2- x.th_0)},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n2_0(x)/2- x.t1_0 - x.t0_0)},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n1_0(x)/2- x.o1_0 - x.o2_0)},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : -x.z1_0 - x.z2_0 - x.z3_0 + n0_0(x)/2)},


    { 'type' : 'ineq',   'fun' : lwe(lambda x : n0_1(x))},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n1_1(x))},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n2_1(x))},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n3_1(x))},

    { 'type' : 'ineq',   'fun' : lwe(lambda x : R1(x))},


    # ========================================================================
    # Constraints in Level 2
    # ========================================================================

    { 'type' : 'ineq',   'fun' : lwe(lambda x : n3_1(x)/2- x.th_1)},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n2_1(x)/2- x.t1_1 - x.t0_1)},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n1_1(x)/2- x.o1_1 - x.o2_1)},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : -x.z1_1 - x.z2_1 - x.z3_1 + n0_1(x)/2)},
    
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n0_2(x))},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n1_2(x))},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n2_2(x))},
    { 'type' : 'ineq',   'fun' : lwe(lambda x : n3_2(x))},

    { 'type' : 'ineq',   'fun' : lwe(lambda x : R2(x))},

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
    start = [ru(0, 0.0009) for _ in range(num_params-4)] + [ru(0.1, 0.9) for _ in range(4)] + [ru(0.005, 0.05)]
    
    bounds=[(0.,0.1)]*(num_params-4)+[(.01,1)]*4 +[(0,0.5)]
    
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