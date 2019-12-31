# Fractional Brownian motion

Fractional Brownian motion generation using Hosking's and Cholesky's methods. Everything is nicely packaged in a python class.

## Definition

Let $H \in ]0,1[$ be a constant. The (1-parameter) fractional Brownian motion (fBm) with Hurst parameter H is the centered Gaussian process $B_H(t) = B_H(t,\omega ), t \in \mathbb{R}, \omega \in \Omega$, satisfying:

$$\mathbb{E}[B_H(s)B_H(t)] = \frac{1}{2}(|s^{2H}|+|t|^{2H}-|s-t|^{2H})$$

Here $\mathbb{E}$ denotes the expectation with respect to the probability measure $\mathbb{P}$ for $\{B_H(t)\}_{t \in \mathbb{R}} = \{B_H(t,\omega); t \in \mathbb{R}, \omega \in \Omega\}$, where $(\Omega, \mathcal{F})$ is a measurable space.

It has stationary increments and is \textit{self-similar} with parameter H: $(B_H(t))_{t \in \mathbb{R}}$ has the same law as $(a^HB_H(t))_{t \in \mathbb{R}}$ for any $a > 0$. Furthermore, sample paths of the fBm have a.s. Hölder regularity $H-\varepsilon$ for any $\varepsilon > 0$.

The fBm has interesting properties. First of all one notices that if $H = \frac{1}{2}$ then the process $B_H(t)$ coincides with the classical Brownian motion. 
If $H > \frac{1}{2}$ then we say that $B_H(t)$ is \textit{persistent}, in the sense that:
$$\sum_{i=1}^\infty Cov[B_H(i+1)-B_H(i), B_H(1)] = \infty$$

This shows that the autocovariance function of the fBm increments decays slowly. This long memory property of the increments when $H > \frac{1}{2}$ motivates its use when modeling persistent phenomena.

If $H < \frac{1}{2}$ then we say that the process is \textit{anti-persistent}, in the sense that:
$$\sum_{i=1}^\infty |Cov[B_H(i+1)-B_H(i), B_H(1)]| < \infty$$

It is worth noting that if $H \neq \frac{1}{2}$ then $B_H$ is not a semi-martingale, so we cannot use the general theory of stochastic calculus for semi-martingales on $B_H$. We thus need a new stochastic calculus for the fBm.

*iPyNB soon with examples and some math*
