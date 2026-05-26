from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class PortfolioAllocation:
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    score: float
    method: str = "risk_return_policy"


class RiskReturnPolicyAgent:
    def __init__(self, risk_aversion: float = 2.0):
        self.risk_aversion = risk_aversion

    def suggest_allocation(self, prices_by_symbol: Dict[str, pd.Series]) -> PortfolioAllocation:
        returns = self._prepare_returns(prices_by_symbol)
        if returns.empty:
            return PortfolioAllocation({}, 0.0, 0.0, 0.0)

        mean_returns = returns.mean() * 252
        covariance = returns.cov() * 252
        raw_scores = mean_returns - self.risk_aversion * np.diag(covariance)
        raw_scores = raw_scores.clip(lower=0)

        if raw_scores.sum() == 0:
            weights = pd.Series(1 / len(raw_scores), index=raw_scores.index)
        else:
            weights = raw_scores / raw_scores.sum()

        expected_return = float((weights * mean_returns).sum())
        expected_volatility = float(np.sqrt(weights.T @ covariance @ weights))
        score = expected_return - self.risk_aversion * expected_volatility

        return PortfolioAllocation(
            weights={symbol: float(weight) for symbol, weight in weights.items()},
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            score=float(score),
        )

    def _prepare_returns(self, prices_by_symbol: Dict[str, pd.Series]) -> pd.DataFrame:
        clean = {}
        for symbol, series in prices_by_symbol.items():
            if series is None or series.empty:
                continue
            clean[symbol] = pd.to_numeric(series, errors="coerce")
        if not clean:
            return pd.DataFrame()
        prices = pd.DataFrame(clean).ffill().dropna()
        return prices.pct_change().dropna()


class PortfolioEnvironment:
    def __init__(self, symbols: Iterable[str], initial_cash: float = 100000.0):
        self.symbols = list(symbols)
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.weights = {symbol: 0.0 for symbol in self.symbols}

    def reset(self):
        self.cash = self.initial_cash
        self.weights = {symbol: 0.0 for symbol in self.symbols}
        return self.weights.copy()

    def step(self, allocation: Dict[str, float], realized_returns: Dict[str, float]):
        self.weights = _normalize_weights(allocation, self.symbols)
        portfolio_return = sum(self.weights[s] * realized_returns.get(s, 0.0) for s in self.symbols)
        self.cash *= 1 + portfolio_return
        reward = portfolio_return
        return self.weights.copy(), reward, {"equity": self.cash}


def _normalize_weights(allocation: Dict[str, float], symbols: list[str]) -> Dict[str, float]:
    values = {symbol: max(float(allocation.get(symbol, 0.0)), 0.0) for symbol in symbols}
    total = sum(values.values())
    if total == 0 and symbols:
        return {symbol: 1 / len(symbols) for symbol in symbols}
    return {symbol: value / total for symbol, value in values.items()}
