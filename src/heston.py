# -*- coding: utf-8 -*-
"""
heston.py - Definition of classic Heston model based on the cleaned
bitcoin option data.
"""
import pandas as pd
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.integrate import quad
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

    Notes on numerical stability:
    - Use j-specific \f$b_j\f$ inside \f$d_j, g_j, C_j, D_j\f$:
      \f$b_1=\kappa-\rho\sigma,\, b_2=\kappa\f$.
    - Spot-centered CF includes \f$(r-q)iuT\f$ and \f$iu\ln S_0\f$;
      integral kernel uses \f$\ln K\f$.
    - Integrate with an adaptive quadrature (quad) over \f$u\f$ with
      near-zero avoidance to reduce oscillatory error at short maturities.
    """

    def __init__(self, kappa: float, theta: float, sigma: float,
                 rho: float, v0: float, r: float = 0.0,
                 q: float = 0.0, debug: bool = False) -> None:
        """
        @brief Constructor for HestonModel.
        @param kappa Mean reversion speed.
        @param theta Long-term variance mean.
        @param sigma Volatility of volatility.
        @param rho Correlation between asset and variance.
        @param v0 Initial variance.
        @param r Risk-free rate.
        @param q Dividend yield or convenience yield.
        @param debug Enable verbose debugging prints.
        @return None
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.r = r
        self.q = q
        self.debug = debug

    def _cf_spot_centered(self, u: complex, T: float,
                          S0: float, j: int) -> complex:
        """
        @brief Heston characteristic function under j (spot-centered).
        @param u Complex argument.
        @param T Time to maturity.
        @param S0 Spot price.
        @param j Index (1: share measure, 2: risk-neutral).
        @return Complex CF value.
        @details
        Spot-centered CF includes drift \f$(r-q)iuT\f$ and \f$iu\ln S_0\f$.
        Use the Little Heston Trap with j-specific \f$b_j\f$ in
        \f$d_j, g_j, C_j, D_j\f$:
        \f[
        b_1 = \kappa - \rho\sigma,\quad b_2 = \kappa
        \f]
        """
        kappa, theta, sigma, rho, v0, r, q = (
            self.kappa, self.theta, self.sigma,
            self.rho, self.v0, self.r, self.q
        )
        iu = 1j * u
        b_j = kappa - rho * sigma if j == 1 else kappa

        # d_j uses b_j (trap form)
        d = np.sqrt((rho * sigma * iu - b_j)**2 +
                    sigma * sigma * (iu + u * u))
        # enforce Re(d) > 0
        if np.real(d) < 0:
            d = -d

        num = b_j - rho * sigma * iu - d
        den = b_j - rho * sigma * iu + d
        g = num / den
        # clip g slightly away from 1 to avoid log blow-up
        eps = 1e-12
        if np.abs(g) > 1 - eps:
            g = (1 - eps) * g / np.abs(g)

        exp_dt = np.exp(-d * T)
        C = (r - q) * iu * T + (kappa * theta / (sigma * sigma)) * (
            num * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g))
        )
        D = (num / (sigma * sigma)) * (
            (1.0 - exp_dt) / (1.0 - g * exp_dt)
        )

        return np.exp(C + D * v0 + iu * np.log(S0))

    def _P1_integrand(self, u: float, T: float,
                      S0: float, K: float) -> float:
        """
        @brief Real part integrand for P1.
        @param u Integration variable (>0).
        @param T Time to maturity.
        @param S0 Spot price.
        @param K Strike price.
        @return Real integrand value.
        @details
        \f[
        \mathrm{Integrand}_{P1} =
        \Re\left(
        \frac{e^{-iu\ln K}\,\phi_1(u-i)}{iu\,\phi_1(-i)}
        \right)
        \f]
        """
        if u == 0.0:
            u = 1e-12
        phi_num = self._cf_spot_centered(u - 1j, T, S0, 1)
        phi_den = self._cf_spot_centered(-1j, T, S0, 1)
        val = np.exp(-1j * u * np.log(K)) * (phi_num / (1j * u * phi_den))
        return float(np.real(val))

    def _P2_integrand(self, u: float, T: float,
                      S0: float, K: float) -> float:
        """
        @brief Real part integrand for P2.
        @param u Integration variable (>0).
        @param T Time to maturity.
        @param S0 Spot price.
        @param K Strike price.
        @return Real integrand value.
        @details
        \f[
        \mathrm{Integrand}_{P2} =
        \Re\left(
        \frac{e^{-iu\ln K}\,\phi_2(u)}{iu}
        \right)
        \f]
        """
        if u == 0.0:
            u = 1e-12
        phi = self._cf_spot_centered(u, T, S0, 2)
        val = np.exp(-1j * u * np.log(K)) * (phi / (1j * u))
        return float(np.real(val))

    def _probability(self, S0: float, K: float, T: float,
                     j: int) -> float:
        """
        @brief Computes risk-neutral probability P1 or P2.
        @param S0 Spot price.
        @param K Strike price.
        @param T Time to maturity.
        @param j Index (1 or 2).
        @return Probability value.
        @details
        Spot-centered integrals (Heston 1993):
        \f[
        P_1 = \frac{1}{2} + \frac{1}{\pi}\int_0^\infty
        \Re\left(\frac{e^{-iu\ln K}\phi_1(u-i)}{iu\phi_1(-i)}\right)du
        \f]
        \f[
        P_2 = \frac{1}{2} + \frac{1}{\pi}\int_0^\infty
        \Re\left(\frac{e^{-iu\ln K}\phi_2(u)}{iu}\right)du
        \f]
        We evaluate with adaptive quadrature and a dynamic upper bound
        \f$ u_{\max} \sim \frac{c}{\sqrt{T}} \f$ to improve convergence
        at short maturities.
        """
        # Dynamic u_max based on maturity and vol-of-vol
        base = 80.0
        scale = 1.0 / np.sqrt(max(T, 1e-6))
        u_max = min(400.0, base * scale)

        # Integrate in two parts to avoid u=0 singularity
        # [1e-8, u_max] with absolute/relative tolerances
        if j == 1:
            integrand = lambda uu: self._P1_integrand(uu, T, S0, K)
        else:
            integrand = lambda uu: self._P2_integrand(uu, T, S0, K)

        val, err = quad(integrand, 1e-8, u_max,
                        epsabs=1e-8, epsrel=1e-7, limit=200)
        P = 0.5 + (1.0 / np.pi) * val

        if self.debug:
            print("[DEBUG] P{}: {:.6f}, int {:.6e}, err {:.2e}, "
                  "u_max {:.2f}, lnK={:.6f}, S0={:.2f}"
                  .format(j, P, val, err, u_max,
                          float(np.log(K)), float(S0)))
        if not np.isfinite(P):
            P = 0.5
        return float(P)

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
        \f[
        C = S_0 e^{-qT}\,P_1 - K e^{-rT}\,P_2
        \f]
        Put price via parity:
        \f[
        P = C - S_0 e^{-qT} + K e^{-rT}
        \f]
        """
        P1 = self._probability(S0, K, T, 1)
        P2 = self._probability(S0, K, T, 2)
        call = S0 * np.exp(-self.q * T) * P1 - \
            K * np.exp(-self.r * T) * P2
        if self.debug:
            print(("[DEBUG] Price inputs: P1={:.6f}, P2={:.6f}, "
                  + "S0e^(-qT)={:.2f}, Ke^(-rT)={:.2f}")
                  .format(P1, P2,
                          S0 * np.exp(-self.q * T),
                          K * np.exp(-self.r * T)))
        if option_type == "call":
            return float(call)
        else:
            return float(call - S0 * np.exp(-self.q * T) +
                         K * np.exp(-self.r * T))

    def calibrate(self, option_data: list[dict],
                  S0: float) -> OptimizeResult:
        """
        @brief Calibrates Heston parameters to market data.
        @param option_data List of dicts with keys:
               K, T, market_price, type.
        @param S0 Spot price.
        @return Optimization result object.
        @details
        Minimize weighted squared price error with Feller penalty:
        \f[
        \mathrm{Err} =
        \sum_i w_i\,(P_{model}-P_{mkt})^2
        + \lambda\,\max(0,\,\sigma^2 - 2\kappa\theta)
        \f]
        Weights \f$w_i\f$ reduce sensitivity to noisy bids/asks.
        """
        # Build weights if available; else uniform
        weights = []
        for opt in option_data:
            w = 1.0
            if "spread_pct" in opt and np.isfinite(opt["spread_pct"]):
                w = 1.0 / (max(opt["spread_pct"], 1e-3)**2)
            weights.append(w)

        def objective(params):
            self.kappa, self.theta, self.sigma, self.rho, self.v0 = params
            feller_violation = max(
                0.0, self.sigma**2 - 2.0*self.kappa*self.theta
            )
            penalty = 1e4 * feller_violation

            errs = []
            for w, opt in zip(weights, option_data):
                mp = self.option_price(S0, opt["K"], opt["T"], opt["type"])
                errs.append(w * (mp - opt["market_price"])**2)

            err = float(np.sum(errs)) + penalty
            if self.debug:
                mae = np.mean(np.sqrt(np.asarray(errs))) if errs else 0.0
                print("[DEBUG] Params kappa={:.4f}, theta={:.4f}, "
                      "sigma={:.4f}, rho={:.4f}, v0={:.4f} | "
                      "penalty={:.2e}, mae={:.2f}"
                      .format(self.kappa, self.theta, self.sigma,
                              self.rho, self.v0, penalty, mae))
            return err

        # Tighter bounds to avoid extremes
        bounds = [(1e-3, 10.0), (1e-6, 1.0), (1e-3, 2.0),
                  (-0.95, 0.95), (1e-6, 1.0)]
        initial = [1.2, 0.05, 0.6, -0.4, 0.05]
        result = minimize(objective, initial, bounds=bounds,
                          method="L-BFGS-B", options={"maxiter": 200})
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
            "type": row["type"],
            "spread_pct": float(row["spread_pct"])
            if "spread_pct" in row and
            np.isfinite(row["spread_pct"]) else None
        })
    return option_data, float(df["S0"].iloc[0])


def main(path: str = ""):
    if not path:
        path = "data/deribit_btc_options_clean.csv"

    # Parse data
    option_data, S0 = parse_option_data(path)

    # Initialize Heston model with arbitrary parameters
    model = HestonModel(kappa=1.0, theta=0.04, sigma=0.3,
                        rho=-0.5, v0=0.04, r=0.01, q=0.0,
                        debug=False)

    # Calibrate model based on the first 10 options
    result = model.calibrate(option_data[:10], S0)
    print("Calibration result:", result.x)

    # Test pricing for test option
    for i in (0, 5):
        test_opt = option_data[i]
        price = model.option_price(S0, test_opt["K"],
                                test_opt["T"], test_opt["type"])
        print(f"Model price for first option: {price:.4f} USD")
        print(f"Actual market price: {test_opt['market_price']:.4f} USD")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
