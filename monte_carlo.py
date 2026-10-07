# -*- coding: utf-8 -*-

'''

Monte Carlo Options Pricer

Option 1 - European call, priced with terminal prices simulation and validated
           against the analytical solution of the Black-Scholes equation

Option 2 - Path-Dependant Asian call, priced with full path simulation and
           validated against the analytical solution of the geometric average
           Asian call option. The option being priced utilises the arithmetic
           average, which has no analytical solution

'''

import numpy as np

from scipy.stats import norm

import matplotlib.pyplot as plt

# Analytical solutions for comparison

def black_scholes_call(S0, K, r, sigma, T):

    '''

    Returns the analytical solution to the Black-Scholes equation, used for
    comparison with estimates

    '''

    d1 = (np.log(S0/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def geometric_asian_call(S0, K, r, sigma, T, n_steps):

    '''

    Returns the analytical solution to the geometric-average Asian call option,
    used for comparison with estimates

    '''

    sigma_g = sigma * np.sqrt((2 * n_steps + 1) / (6 * (n_steps + 1)))

    mu_g = 0.5 * sigma_g**2 + (r - 0.5 * sigma**2) * (n_steps + 1) / (2 * n_steps)

    d1 = (np.log(S0/K) + (mu_g + 0.5 * sigma_g**2) * T) / (sigma_g * np.sqrt(T))

    d2 = d1 - sigma_g * np.sqrt(T)


    return np.exp(-r * T) * (S0 * np.exp(mu_g * T) * norm.cdf(d1) - K * norm.cdf(d2))

# Simulating geometric Brownian motion

def simulate_terminal_prices(S0, r, sigma, T, n_paths, rng):

    '''

    Simulates the terminal prices of the European call otpion from the analytical
    GBM solution

    '''

    Z = rng.standard_normal(n_paths)

    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

def simulate_paths_from_Z(S0, r, sigma, T, n_steps, Z,):

    '''

    Simulates the full GBM price path from an externally supplied array of
    normal draws.

    '''

    dt = T / n_steps

    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    log_paths = np.cumsum(increments, axis = 1)

    paths = S0 * np.exp(log_paths)

    return np.hstack([np.full((Z.shape[0], 1), S0), paths])

def simulate_paths(S0, r, sigma, T, n_steps, n_paths, rng):

    '''

    Returns a full GBM price path by generating an array of normal draws and
    then passing it to simulate_paths_from_Z.

    '''

    Z = rng.standard_normal((n_paths, n_steps))

    return simulate_paths_from_Z(S0, r, sigma, T, n_steps, Z)

# Calculating payoffs

def european_call_payoff(ST, K):

    '''

    Calculates the payoff of a european call option

    '''

    return np.maximum(ST - K, 0.0)

def asian_call_payoff(paths, K, average_type):

    '''

    Calulates the payoff of an asian call option for arithemtic and geometric
    averages

    '''

    monitored = paths[:, 1:]

    if average_type == 'arithmetic':

        avg = monitored.mean(axis=1)

    elif average_type == 'geometric':

        avg = np.exp(np.log(monitored).mean(axis = 1))

    else:
        raise ValueError('average_type must be "arithmetic" or "geometric"')

    return np.maximum(avg - K, 0.0)

# Monte Carlo pricers

def mc_european_call(S0, K, r, sigma, T, n_paths, rng):

    '''

    Monte Carlo pricer for a European call option

    Simulates terminal prices, computes the discounted payoff on each, returns
    the price estimate and its standard error

    '''

    ST = simulate_terminal_prices(S0, r, sigma, T, n_paths, rng)

    discounted = np.exp(-r * T) * european_call_payoff(ST, K)

    price = discounted.mean()

    std_error = discounted.std(ddof = 1) / np.sqrt(n_paths)

    return price, std_error

def mc_asian_call(S0, K, r, sigma, T, n_steps, n_paths, rng, average_type):

    '''

    Monte Carlo pricer for an Asian call option
    
    Simulates the full price paths, computes the discounted payoff on each paths
    average price, returns the price estimate and its standard error

    '''

    paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths, rng)

    discounted = np.exp(-r * T) * asian_call_payoff(paths, K, average_type)

    price = discounted.mean()

    std_error = discounted.std(ddof=1) / np.sqrt(n_paths)

    return price, std_error

# Methods of variance reduction

# Method 1: Antithetic variates

def mc_asian_call_antithetic(S0, K, r, sigma, T, n_steps, n_pairs, rng, average_type):

    '''
    
    Antithetic variates - For each random Z, also simulate the mirror path with -Z
                          and then average the final payoffs. Helps to cancel out
                          noise in the simulation

    '''

    Z = rng.standard_normal((n_pairs, n_steps))

    paths_pos = simulate_paths_from_Z(S0, r, sigma, T, n_steps, Z)

    paths_neg = simulate_paths_from_Z(S0, r, sigma, T, n_steps, -Z)

    payoff_pos = asian_call_payoff(paths_pos, K, average_type=average_type)

    payoff_neg = asian_call_payoff(paths_neg, K, average_type=average_type)

    # Averaging each antithetic pair before averaging across pairs

    payoff_paired = 0.5 * (payoff_pos + payoff_neg)

    discounted = np.exp(-r * T) * payoff_paired

    price = discounted.mean()

    std_error = discounted.std(ddof=1) / np.sqrt(n_pairs)

    return price, std_error

# Method 2: Control variates

def mc_asian_call_control(S0, K, r, sigma, T, n_steps, n_paths, rng):

    '''

    Control variates - The arithmetic and geometric payoffs are calculated from the
                       the same simulated paths, but the geometric average has a
                       known analytical solution. We can use the error on the
                       geometric estimate to inform the arithmetic estimate

    '''

    paths = simulate_paths(S0, r, sigma, T, n_steps, n_paths, rng)

    arith_payoff = asian_call_payoff(paths, K, average_type = 'arithmetic')

    geo_payoff = asian_call_payoff(paths, K, average_type = 'geometric')

    discounted_arith = np.exp(-r * T) * arith_payoff

    discounted_geo = np.exp(-r * T) * geo_payoff

    geo_exact = geometric_asian_call(S0, K, r, sigma, T, n_steps)

    beta = (np.cov(discounted_arith, discounted_geo, ddof = 1)[0, 1]
            / np.var(discounted_geo, ddof = 1))

    adjusted = discounted_arith - beta * (discounted_geo - geo_exact)

    price = adjusted.mean()

    std_error = adjusted.std(ddof = 1) / np.sqrt(n_paths)

    return price, std_error

# Convergence of MC European price to Black-Scholes

def european_convergence(S0, K, r, sigma, T, rng):

    '''

    Plots the convergence of the monte carlo simulation to the analytical
    solution of the Black-Scholes equation for the european call option

    '''

    print('=== European call option - Monte Carlo vs Black-Scholes ===\n')

    analytical_price = black_scholes_call(S0, K, r, sigma, T)

    print(f'   Analytical Black-Scholes price: {analytical_price:.4f}\n')

    path_counts = np.logspace(2, 6, 15).astype(int)

    prices, errors = [], []

    for n in path_counts:

        price, std_error = mc_european_call(S0, K, r, sigma, T, n, rng)

        prices.append(price)

        errors.append(1.96 * std_error)

        print(f'   n_paths = {n:>8}   price={price:.4f}   95% CI = +/-{1.96*std_error:.4f}')

    fig, ax = plt.subplots()

    ax.errorbar(path_counts, prices, yerr=errors, fmt='x-', capsize=3, label='Monte Carlo (95% CI)')

    ax.axhline(analytical_price, color='red', linestyle='--', label='Black-Scholes analytical')

    ax.set_xscale('log')

    ax.set_xlabel('Number of simulated paths')

    ax.set_ylabel('Option price')

    ax.set_title('Monte Carlo European call price convergence')

    ax.legend()

    ax.grid(alpha = 0.3)

    fig.tight_layout()

    fig.savefig('mc_european_convergence.png', dpi = 150)

# Comparison of geometric vs arithmetic avg for Asian option

def asian_comparison(S0, K, r, sigma, T, n_steps, n_paths, rng):

    '''

    Outputs a numerical comparison of the arithemtic estimate and the geometric
    analytical solution. Also confirms that the arithmetic price is greater than
    the geometric price

    '''

    print('=== Asian call option - Geometric (Analytical) vs Arithmetic (MC) ===\n')

    geo_price_mc, geo_se = mc_asian_call(S0, K, r, sigma, T, n_steps,
                                         n_paths, rng, average_type='geometric')

    geo_price_exact = geometric_asian_call(S0, K, r, sigma, T, n_steps)

    print('Geometric-average Asian call:')
    print(f'   Monte Carlo price: {geo_price_mc:.4f}   (95% CI +/- {1.96 * geo_se:.4f})')
    print(f'   Closed-form price: {geo_price_exact:.4f}')
    print(f'   Difference       : {abs(geo_price_mc - geo_price_exact):.4f}\n')

    arith_price_mc, arith_se = mc_asian_call(S0, K, r, sigma, T, n_steps,
                                             n_paths, rng, average_type='arithmetic')

    print('Arithmetic-average Asian call (determined via MC):')
    print(f'   Monte Carlo price: {arith_price_mc:.4f}  (95% CI +/- {1.96 * arith_se:.4f})\n')

    print(f'Arithmetic > Geometric (expected by AM - GM): '
          f'{arith_price_mc:.4f} > {geo_price_mc:.4f} -> {arith_price_mc > geo_price_mc}\n')

# Variance reduction applied

def variance_reduction(S0, K, r, sigma, T, n_steps, rng):

    '''

    Applies the methods of antithetic and control variates to reduce the error
    on the simulated Asian price

    '''

    print('=== Variance reduction methods ===\n')

    n_paths = 200_000

    print(f'Methods are compared at a fixed number of {n_paths:,} simulated paths\n')

    plain_price, plain_se = mc_asian_call(S0, K, r, sigma, T, n_steps,
                                              n_paths, rng, average_type = 'arithmetic')

    print(f'   Plain MC               price={plain_price:.4f}  std_error={plain_se:.5f}\n')

    anti_price, anti_se = mc_asian_call_antithetic(S0, K, r, sigma, T, n_steps,
                                                   n_paths // 2, rng, average_type = 'arithmetic')

    print(f'   Antithetic variate     price={anti_price:.4f}  std_error={anti_se:.5f}')

    print(f' {100 * (1 - anti_se/plain_se):.1f}% lower standard error\n')

    cv_price, cv_se = mc_asian_call_control(S0, K, r, sigma, T, n_steps, n_paths, rng)

    print(f'   Control variate        price={cv_price:.4f}  std_error={cv_se:.5f}')

    print(f' {100 * (1 - cv_se/plain_se):.1f}% lower standard error')

    # Apply for a large range of path counts (for plotting)

    print('\n Repeating for a range of path counts for the comparison plot\n')

    path_counts = np.logspace(3, 5.5, 10).astype(int)

    plain_errors = []

    anti_errors = []

    cv_errors = []

    for n in path_counts:

        se_plain = mc_asian_call(S0, K, r, sigma, T, n_steps,
                                 n, rng, average_type = 'arithmetic')[1]

        se_anti = mc_asian_call_antithetic(S0, K, r, sigma, T, n_steps,
                                           n // 2, rng, average_type = 'arithmetic')[1]

        se_cv = mc_asian_call_control(S0, K, r, sigma, T, n_steps, n, rng)[1]

        plain_errors.append(se_plain)

        anti_errors.append(se_anti)

        cv_errors.append(se_cv)

        print(f'  n_paths = {n:>7},  plain={se_plain:.5f}  antithetic={se_anti:.5f}  cv = {se_cv:.5f}')

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(path_counts, plain_errors, 'o-', label = 'Plain Monte Carlo')

    ax.plot(path_counts, anti_errors, 's-', label = 'Antithetic variates')

    ax.plot(path_counts, cv_errors, '^-', label = 'Control variates')

    ax.set_xscale('log')

    ax.set_yscale('log')

    ax.set_xlabel('Number of simulated paths')

    ax.set_ylabel('Standard error on estimated price')

    ax.set_title('Variance reduction for arithmetic Asian call option')

    ax.legend()
    
    ax.grid(alpha = 0.3)

    fig.tight_layout()

    fig.savefig('variance_reduction.png', dpi=150)

# Plotting a sample of paths for illustrative purposes

def sample_paths_plot(S0, r, sigma, T, n_steps, n_plots, rng, n_sample):

    '''

    Plots a sample of simulated paths. The paths with the most extreme
    final prices are highlighted

    '''

    paths = simulate_paths(S0, r, sigma, T, n_steps, n_sample, rng)

    final_prices = paths[:, -1]

    price_max = np.argmax(final_prices)

    price_min = np.argmin(final_prices)

    remaining_paths = np.setdiff1d(np.arange(n_sample), [price_max, price_min])

    n_random = n_plots - 2

    random_paths = rng.choice(remaining_paths, size = n_random, replace = False)

    times = np.linspace(0, T, n_steps + 1)

    fig, ax = plt.subplots(figsize = (8, 5))

    ax.plot(times, paths[random_paths].T, linewidth = 0.6, alpha = 0.4, color = 'blue')

    ax.plot(times, paths[price_max], linewidth = 2, color = 'green',
            label = f'Max final price ({final_prices[price_max]:.2f})')

    ax.plot(times, paths[price_min], linewidth = 2, color = 'red',
            label = f'Min final price ({final_prices[price_min]:.2f})')

    ax.axhline(S0, color = 'black', linestyle = '--', linewidth = 1, label = 'S0')

    ax.set_xlabel('Time (yrs)')

    ax.set_ylabel('Simulated stock price')

    ax.set_title(f'{n_plots} simulated paths (extremes from a pool of {n_sample:,} highlighted)')

    ax.legend()
    
    ax.grid(alpha = 0.1)

    fig.tight_layout()

    fig.savefig('sample_paths.png', dpi = 150)

# Main

def main():

    '''

    Main function

    '''

    # Shared parameters

    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0

    n_steps = 252

    n_paths_asian = 200000

    rng = np.random.default_rng(42)

    european_convergence(S0, K, r, sigma, T, rng)

    print()

    asian_comparison(S0, K, r, sigma, T, n_steps, n_paths_asian, rng)

    variance_reduction(S0, K, r, sigma, T, n_steps, rng)

    sample_paths_plot(S0, r, sigma, T, n_steps, n_plots=100, rng=rng, n_sample=5000)

if __name__ == '__main__':
    main()
