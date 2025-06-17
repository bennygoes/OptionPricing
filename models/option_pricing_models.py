import numpy as np
from scipy.stats import norm

class BlackScholesModel:
    def __init__(self, r: float):
        self.r = r  # Risk-free interest rate

    def price(self, S: float, K: float, T: float, sigma: float, option_type: str = 'call') -> float:
        d1 = (np.log(S / K) + (self.r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T) + 1e-10)
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        elif option_type == 'put':
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")


class BinomialTreeModel:
    def __init__(self, r: float, steps: int = 100):
        self.r = r
        self.steps = steps

    def price(self, S, K, T, sigma, option_type='call'):
        dt = T / self.steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)

        # Terminal payoffs
        prices = np.zeros(self.steps + 1)
        for i in range(self.steps + 1):
            ST = S * (u ** (self.steps - i)) * (d ** i)
            if option_type == 'call':
                prices[i] = max(ST - K, 0)
            else:
                prices[i] = max(K - ST, 0)

        # Backward induction
        for step in range(self.steps - 1, -1, -1):
            for i in range(step + 1):
                prices[i] = np.exp(-self.r * dt) * (p * prices[i] + (1 - p) * prices[i + 1])

        return prices[0]
    

class SABRModel:
    def __init__(self, alpha, beta, rho, nu, r):
        self.alpha = alpha  # initial vol
        self.beta = beta    # elasticity parameter
        self.rho = rho      # correlation
        self.nu = nu        # vol of vol
        self.r = r          # risk-free rate

    def implied_vol(self, F, K, T):
        if F == K:
            return (self.alpha / F**(1 - self.beta)) * (
                1 + ((1 - self.beta)**2 / 24) * (self.alpha**2 / F**(2 - 2 * self.beta)) * T
            )

        log_fk = np.log(F / K)
        FK_beta = (F * K)**((1 - self.beta) / 2)
        z = (self.nu / self.alpha) * FK_beta * log_fk
        x_z = np.log((np.sqrt(1 - 2 * self.rho * z + z**2) + z - self.rho) / (1 - self.rho))

        return self.alpha * z / x_z / FK_beta * (
            1 + ((1 - self.beta)**2 / 24) * (log_fk**2)
        )

    def price(self, F, K, T, option_type='call'):
        sigma = self.implied_vol(F, K, T)
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            return F * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - F * norm.cdf(-d1)