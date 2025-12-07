# -*- coding: utf-8 -*-
"""
heston.py - Definition of classic Heston model based on the the cleaned
bitcoin option data.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize, OptimizeResult
import sys


class HestonModel:
    """
    @class HestonModel
    @brief Implementation of the Heston stochastic volatility model.

    @details
    The Heston model assumes the asset price S_t and variance v_t follow:
    \f[
    dS_t = (r - q) S_t dt + \sqrt{v_t} S_t dW_t^S
    \f]
    \f[
    dv_t = \kappa(\theta - v_t)dt + \sigma \sqrt{v_t} dW_t^v
    \f]
    with correlation \f$ \rho \f$ between Brownian motions
    \f$ dW_t^S \f$ and \f$ dW_t^v \f$.
    """

    def __init__(self, kappa: float, theta: float, sigma: float,
                 rho: float, v0: float, r: float = 0.0,
                 q: float = 0.0) -> None:
        """
        @brief Constructor for HestonModel.
        @param kappa Mean reversion speed.
        @param theta Long-term variance mean.
        @param sigma Volatility of volatility.
        @param rho Correlation between asset and variance.
        @param v0 Initial variance.
        @param r Risk-free rate.
        @param q Dividend yield or convenience yield.
        @return None
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.r = r
        self.q = q

    def characteristic_function(self, u: complex, T: float,
                                S0: float) -> complex:
        """
        @brief Computes the Heston characteristic function.
        @param u Complex argument.
        @param T Time to maturity.
        @param S0 Initial asset price.
        @return Complex value of characteristic function.
        @details
        The characteristic function is:
        \f[
        \phi(u;T) = \exp(C(T,u) + D(T,u)v_0 + i u \ln S_0)
        \f]
        where C and D are functions of parameters
        \f$ \kappa, \theta, \sigma, \rho, r, q \f$.
        """
        kappa, theta, sigma, rho, v0, r, q = (
            self.kappa, self.theta, self.sigma,
            self.rho, self.v0, self.r, self.q
        )
        d = np.sqrt((rho * sigma * 1j * u - kappa)**2 +
                    (sigma**2) * (1j * u + u**2))
        g = (kappa - rho * sigma * 1j * u - d) / \
            (kappa - rho * sigma * 1j * u + d)
        C = (r - q) * 1j * u * T + (kappa * theta / sigma**2) * (
            (kappa - rho * sigma * 1j * u - d) * T -
            2.0 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )
        D = ((kappa - rho * sigma * 1j * u - d) / sigma**2) * (
            (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        )
        return np.exp(C + D * v0 + 1j * u * np.log(S0))

    def option_price(self, S0: float, K: float, T: float,
                     option_type: str = "call") -> float:
        """
        @brief Computes option price under Heston model.
        @param S0 Spot price.
        @param K Strike price.
        @param T Time to maturity (years).
        @param option_type "call" or "put".
        @return Option price as float.
        @details
        Price is computed via Fourier integration:
        \f[
        P = S_0 e^{-qT} - \frac{K e^{-rT}}{\pi}
        \int_0^\infty Re\left(
        \frac{e^{-iu\ln K}\phi(u-i)}{iu\phi(-i)}
        \right) du
        \f]
        """
        N = 200
        u_max = 100.0
        du = u_max / N
        u = np.linspace(1e-5, u_max, N)

        integrand = np.real(
            np.exp(-1j * u * np.log(K)) *
            self.characteristic_function(u - 1j, T, S0) /
            (1j * u * self.characteristic_function(-1j, T, S0))
        )
        integral = du * np.sum(integrand)

        price = S0 * np.exp(-self.q * T) - \
            (K * np.exp(-self.r * T) / np.pi) * integral
        if option_type == "put":
            price = price - S0 * np.exp(-self.q * T) + \
                K * np.exp(-self.r * T)
        return float(np.real(price))

    def calibrate(self, option_data: list[dict],
                  S0: float) -> OptimizeResult:
        """
        @brief Calibrates Heston parameters to market data.
        @param option_data List of dicts with keys:
               K, T, market_price, type.
        @param S0 Spot price.
        @return Optimization result object.
        @details
        Minimizes squared error:
        \f[
        \text{Error} = \sum_i
        (P_{model}(K_i,T_i) - P_{market,i})^2
        \f]
        """
        def objective(params):
            self.kappa, self.theta, self.sigma, \
                self.rho, self.v0 = params
            error = 0.0
            for opt in option_data:
                model_price = self.option_price(
                    S0, opt["K"], opt["T"], opt["type"]
                )
                error += (model_price - opt["market_price"])**2
            return error

        bounds = [(1e-5, None), (1e-5, None), (1e-5, None),
                  (-0.999, 0.999), (1e-5, None)]
        initial_guess = [1.0, 0.04, 0.3, -0.5, 0.04]
        result = minimize(objective, initial_guess,
                          bounds=bounds, method="L-BFGS-B")
        return result


def parse_option_data(path: str) -> tuple[list[dict], float]:
    """
    @brief Parses option data from CSV file.
    @param path Path to CSV file.
    @return Tuple of (option_data list, spot price).
    """
    df = pd.read_csv(path)
    option_data = []
    for _, row in df.iterrows():
        option_data.append({
            "K": float(row["strike"]),
            "T": float(row["T_years"]),
            "market_price": float(row["market_price_usd"]),
            "type": row["type"]
        })
    return option_data, float(df["S0"].iloc[0])


def main(path: str = ""):
    if not path:
        path = "data/deribit_btc_options_clean.csv"

    # Parse data
    option_data, S0 = parse_option_data(path)

    # Initialize Heston model with arbitrary parameters
    model = HestonModel(kappa=1.0, theta=0.04, sigma=0.3,
                        rho=-0.5, v0=0.04, r=0.01, q=0.0)

    # Test pricing for first option
    first_opt = option_data[0]
    price = model.option_price(S0, first_opt["K"],
                               first_opt["T"], first_opt["type"])
    print(f"Model price for first option: {price:.4f} USD")

    # Calibrate model based on the first 10 options
    result = model.calibrate(option_data[:10], S0)
    print("Calibration result:", result.x)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
