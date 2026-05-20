# Improved Polynomial-Memory Attacks on Small-Secret LWE

This repository contains the implementation and experimental code accompanying the paper:

    Improved Polynomial-Memory Attacks on Small-Secret LWE

The repository provides:
- optimization scripts for the asymptotic complexity analysis,
- experimental validation code for the heuristic assumptions.
- Data for optimal paramters

-------------------------------------------------------------------------------

# Overview

This work improves the Memory-Efficient Attacks on Small LWE Keys introduced by Esser et al. (ASIACRYPT 2023).

The main contributions include:
- an improved nested collision-search framework,
- the new Shifted-3 instantiation,
- and the unified Hybrid instantiation.

-------------------------------------------------------------------------------

# Requirements

## Python Dependencies

The optimization scripts require:

pip install numpy scipy

## C Compiler

The experiments require a standard C compiler.

For Linux/macOS:

gcc Improved_Nested_2star.c -O3 -o Improved_Nested_2star -lm

-------------------------------------------------------------------------------

# Running the Optimization Scripts

Example:

python3 improved_nested_2star.py

The scripts perform multiple randomized optimization runs and report:
- the best asymptotic complexity exponent,
- and the corresponding optimization parameters.

Key parameters can be modified directly inside the scripts:

w = 0.667          # Relative weight

TOTAL_RUNS = 1000 # Number of optimization runs

Num_cores = 20    # Parallel processes

-------------------------------------------------------------------------------

# Running the Experimental Validation

Compile:

gcc Improved_Nested_2star.c -O3 -o Improved_Nested_2star -lm

Run:

./Improved_Nested_2star

The experiment validates:
- projected uniqueness heuristics,
- collision estimates,
- middle-layer event probabilities,
- and distinguished-collision statistics.

-------------------------------------------------------------------------------

# Reproducibility

The optimization routines are non-convex and do not guarantee convergence to a global optimum. To improve confidence in the reported values:
- multiple randomized initializations are used,
- optimization is repeated many times,
- and the best solution across all runs is reported.

The experimental validation uses enumeration on toy-scale instances to verify the heuristic assumptions underlying the asymptotic analysis.


-------------------------------------------------------------------------------

# Acknowledgements

This work builds upon the nested collision-search framework introduced by:

- Esser, Girme, Mukherjee, and Sarkar,
  Memory-Efficient Attacks on Small LWE Keys,
  ASIACRYPT 2023.

