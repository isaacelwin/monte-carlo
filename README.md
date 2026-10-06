# Monte-Carlo Options Pricer

A Monte-Carlo program for pricing European and Asian call options, extended with applications of variance reduction techniques and 
validation against analytical solutions.

## Background

This project prices two seperate options:

- **European call**: Priced with terminal price simulation, and validated against exact solution given by the Black-Scholes equation

- **Asian call**: Priced with full path simulation as the payoff depends on the option's average price over its lifetime.
  There exists no exact solution to the real arithmetic-average version, and so the analytical solution to the
  geometric-average version is used for validation (and later implemented as a control variate)

Additionally, two variance reduction techniques (antithetic variates and control variates) are implemented and compared with the plain 
Monte-Carlo simulation

## Results

- **European call convergence**: The Monte-Carlo price (10.4577) converges to the Black-Scholes analytical price (10.4506) after
  1,000,000 simulated paths

- **Geometric Asian call validation**: Monte-Carlo price is consistent with the analytical solution within statistical noise

- **Arithmetic > Geometric**: Confirmed by AM-GM, as expected for any set of prices

- **Variance reduction**: At 200,000 paths, antithetic variates reduce the standard error by ~30%, and control variates (utilising
  the correlation between the geometric and arithmetic Asian payoffs) reduces the standard error by ~97%

  

## Method


## Limitations

- **Slow convergence**: The 95% confidence interval shrinks as $$\frac{1}{\sqrt N}$$, meaning that to halve the error we need 4x the
  number of simulations

- **Simplified assumptions**: The simulation relies upon the assumption that the option price exactly follows Geometric Brownian Motion
  with constant volatility, which is not the case in the real world


## Setup

```bash
pip install numpy scipy matplotlib
```

## Run

```bash
python monte_carlo.py
```

## Possible Extensions

- Compare against a physics-informed neural network (PINN) solving the Black-Scholes PDE (accuracy vs compute time tradeoffs)
- Compute option Greeks
