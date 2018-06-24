# Mean-Field-Games
On solving Mean Field Games. 

Main Reference: Carmona book: Probabilistic Theory of Mean Field Games with Applications

## Contributions
- We have developed Howard-Bellman Policy Iteration Algorithm for the MFG (Mean Field Games) in continuous time and space models.
- We have developed deep-neural network based algorithms for the approximation of MFG. Our preliminary results are very promising.
- We have developed inverse MFG algorithm that allows to recover cost functions from observed strategies.
- We have preliminary results for major-minor player MFG, where players are not homogeneous. Again we deployed deep-neural network algorithm in this framework which seems to work well.
- We would like to stress out that all these contributions are new and have not been considered in literature so far.

## Code
### PDEs and Deep Learning
- `src/DeepLearning-PDE/pde_sol_approximation.py`: Implementation of algorithm that approximates solution of a PDE from https://arxiv.org/abs/1708.07469
- `src/DeepLearning-PDE/pde_sol_approximation_grad.py` : Implementation of algorithm that approximates soluton of a PDE using Deep Learning from https://arxiv.org/pdf/1706.04702.pdf

### Inverse Reinforcement Learning
Task:
_Given measurements of an agent’s behaviour over time, determine the cost function being optimized._

Hypotheses:
- Assume that we observe the behavior of an agent, through the value function $v^⋆(t,x)$ or the policy function $\alpha(x)$, at any state and time points.
- Moreover, we assume that we can find the value function(or policy) for any cost function of our choice. (following the algorithm developed in `src/Stochasti-Control-LQR/policy_improvement.py`
- In the IRL framework, we seek to find the values $b^f$ , $c^f$ and $\gamma$ that determine the cost function of the agent.

Code:
- `src/Inverse-Reinforcement-Learning/LQR_IRL.py`

### Mean Field Games - Law improvement code
- `src/MFG/MFG_LQR_major_minor_players.py`: Mean Field Game in the LQR setting with a major player and N minor players
- `src/MFG/MFG_policy_improvement.py`: implementation of the MFG solution to find the Nash equilibrium in the Mean Field Linear Quadrtic game. 

### Stochastic Control - Linear Quadratic Regulator problem
- `src/Stochastic-Control-LQR/flocking.py` : flocking model from the Mean Field Book (eq 2.51)
- `src/Stochastic-Control-LQR/policy_improvement.py` : Policy improvement of LQR problem without Mean Field Game.

### utils
- `src/utils/first_order_linear_ode_solution.py`: clever implementation of solver of first order linear ode y' = a0(t) + a1(t)y
with terminal condition, where a0(t) a1(t) are given by a sample of points between \[0,T\]





