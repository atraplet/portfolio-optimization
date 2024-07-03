# Portfolio Optimization

Solve a [Markowitz portfolio optimization problem](https://en.wikipedia.org/wiki/Modern_portfolio_theory)
with [ecos4j](https://github.com/atraplet/ecos4j):

A long-only investor wishes to maximize the expected portfolio return given a limit on the portfolio risk

$$
\begin{align*}
& \text{maximize} & & \mu^T x \\
& \text{subject to} & & x^T \Sigma x \leq \sigma^2 \\
& & & \mathbf{1} x = 1 \\
& & & x \geq 0
\end{align*}
$$

where $x$ is the unknown vector of portfolio allocations, $\mu$ is the estimated expected return vector, $\Sigma$ is the
estimated covariance matrix, and $\sigma$ is the given limit on the portfolio risk.
