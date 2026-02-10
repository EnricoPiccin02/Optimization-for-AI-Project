# Chess Engine Design and Optimization with CMA-ES and NEAT

This repository documents the design, implementation, and optimisation of a lightweight chess engine as part of the **Optimization for AI** master's course. The project explores two distinct optimisation paradigms: **Covariance Matrix Adaptation Evolution Strategy (CMA-ES)** and **NeuroEvolution of Augmenting Topologies (NEAT)**. Both methods are applied to optimise the evaluation function of the chess engine, providing insights into their respective strengths, limitations, and emergent behaviours.

<br/>

## 1. Project Overview

### 1.1 Objective
The primary goal of this project is to study the interaction between **evaluation modelling**, **search algorithms**, and **evolutionary optimisation** in a controlled environment. The chess engine is intentionally minimalist, designed to serve as a well-defined optimisation substrate rather than to compete with state-of-the-art engines. This approach allows for isolating and analysing the statistical and structural properties of the optimisation process.

### 1.2 Key Features
- **Lightweight Chess Engine**: A structurally sound engine relying on an external library for chess rules, enabling focus on optimisation-relevant components.
- **Feature-Based Evaluation**: A compact, fixed-dimensional feature representation of chess positions, designed for expressiveness, low dimensionality, and numerical stability.
- **Two Optimisation Paradigms**:
  - **CMA-ES**: Optimises a linear evaluation function in a continuous parameter space.
  - **NEAT**: Evolves neural network-based evaluation functions, capturing non-linear feature interactions.

<br/>

## 2. Chess Engine Design

### 2.1 Feature Representation
The evaluation function is based on an **11-dimensional feature vector**, capturing key aspects of chess positions. Features are designed to balance **expressiveness** and **sample efficiency**, with all values computed as differences between White and Black to eliminate colour bias. The feature set includes:
- **Material Balance**: Differences in the number of pawns, knights, bishops, rooks, and queens.
- **Positional Features**:
  - **Bishop Pair**: Strategic advantage of owning both bishops.
  - **Mobility**: Difference in the number of legal moves available to each side.
  - **Pawn Structure**: Isolated and doubled pawns as indicators of long-term weaknesses.
  - **King Safety**: Number of opponent attackers targeting the king's square.
  - **Pawn Advancement**: Spatial control and promotion potential.

### 2.2 Evaluation Models
1. **Linear Evaluator**:
   - A weighted sum of features:
     
$$\textbf{score}(\text{board}) = \mathbf{w}^\top \mathbf{f}(\text{board})$$
     
   - Optimised using CMA-ES.
   - Features are normalised to ensure numerical stability and prevent optimisation bias.

2. **Neural Evaluator**:
   - A feedforward neural network with the normalised feature vector as input.
   - Evolved using NEAT to capture non-linear feature interactions.

### 2.3 Move Selection
The engine uses **Negamax search with alpha-beta pruning** for move selection. This approach:
- Exploits the zero-sum nature of chess.
- Reduces the effective branching factor, enabling deeper lookahead at fixed computational cost.

### 2.4 Validation
The engine was validated in a frozen configuration (fixed evaluation parameters) through:
- **Self-Play Symmetry**: Ensuring identical engines converge to an average score of $0.5$.
- **Absence of Colour Bias**: Verifying fairness across many games.
- **Controlled Stochasticity**: Randomised tie-breaking to prevent pathological loops.
- **Fitness Signal Suitability**: Balancing noise and informativeness for evolutionary optimisation.

The engine validation through $50$ self-play games produced the following results:

| Metric                               | Value   |
|:-------------------------------------|:-------:|
| Games Played                         | $50$    |
| $1-0$ Outcomes                       | $24$    |
| $0-1$ Outcomes                       | $23$    |
| $\tfrac{1}{2}-\tfrac{1}{2}$ Outcomes | $3$     |
| Average Score                        | $0.510$ |
| Score Standard Deviation             | $0.485$ |

<br/>

## 3. CMA-ES Optimisation

### 3.1 Overview
CMA-ES is a state-of-the-art algorithm for continuous, noisy, black-box optimisation. It is particularly well suited for tuning the linear evaluation function.

### 3.2 Fitness Evaluation
Fitness is defined as the **expected score** of the engine against a fixed baseline opponent, estimated via Monte Carlo self-play. Outcomes are mapped to numerical scores (win = $1$, draw = $0.5$, loss = $0$), and fitness is computed as the mean score across games.

### 3.3 Experimental Protocol
- **Search Space**: $11$-dimensional continuous space (feature weights).
- **Population Size**: Given $d = 11$

$$\lambda = 4 + \lfloor 3 \log(d) \rfloor$$

- **Champion Re-Evaluation**: Periodic re-evaluation of the best solution with a larger evaluation budget to reduce noise.

### 3.4 Results
CMA-ES demonstrated rapid early improvements, followed by convergence to strong evaluators. Final champions achieved win rates significantly above baseline expectations.

<img width="1702" height="1361" alt="cma-es-convergence" src="https://github.com/user-attachments/assets/e7a29f17-eff9-4b1f-80c7-ae9c97deee28" />

### 3.5 Champion Evaluation
The final CMA-ES champion was evaluated over $200$ games, achieving the following results:

| Metric            | Value   |
|:------------------|:-------:|
| Games Played      | $200$   |
| Win Rate          | $0.723$ |
| Baseline Expected | $0.500$ |

<br/>

## 4. NEAT Optimisation

### 4.1 Overview
NEAT evolves populations of neural networks, enabling non-linear evaluation functions and population-level diversity. Unlike CMA-ES, NEAT operates in a dynamic search space, allowing for representational flexibility.

### 4.2 Genome Representation
- **Node Genes**: Specify neurons and their types.
- **Connection Genes**: Specify weighted synaptic links, with innovation numbers ensuring consistent alignment during crossover.

### 4.3 Evolutionary Dynamics
- **Selection**: Top $50\%$ of genomes are retained each generation.
- **Crossover**: Aligns matching genes via innovation numbers.
- **Mutation**: Perturbs connection weights with Gaussian noise. Structural mutations are disabled to isolate the effect of weight adaptation.

### 4.4 Results
NEAT exhibited steady fitness improvements and convergence towards high-performing evaluators. The best individuals occasionally achieved perfect scores against the baseline, demonstrating robust optimisation.

<img width="1702" height="1361" alt="neat-convergence" src="https://github.com/user-attachments/assets/940ef502-2cae-4492-9f26-36781966ac5a" />

### 4.5 Champion Evaluation
The final NEAT champion was evaluated over $200$ games, achieving the following results:

| Metric            | Value |
|-------------------|:-------:|
| Games Played      | $200$   |
| Win Rate          | $0.838$ |
| Baseline Expected | $0.500$ |

<br/>

## 5. Comparative Discussion

### 5.1 CMA-ES vs. NEAT
- **CMA-ES**:
  - Operates in a fixed, low-dimensional continuous space.
  - Excels at exploiting smooth fitness landscapes through covariance adaptation.
  - Highly sample-efficient, but limited to linear or near-linear models.
- **NEAT**:
  - Introduces representational flexibility through non-linear neural evaluators.
  - Captures complex feature interactions but is computationally more expensive.

Both methods successfully optimised the chess engine, highlighting complementary trade-offs between **model bias**, **search efficiency**, and **expressive capacity**.

<br/>

## 6. Additional Contribution: Qualitative Analysis

In addition to quantitative convergence analysis, we include a **qualitative head-to-head match** between the best-performing CMA-ES and NEAT agents. This visualisation is not used as statistical evidence but as an interpretability tool to highlight behavioural differences induced by optimisation bias and representation choice.

While both agents optimise the same objective under comparable computational budgets, their play styles differ markedly, illustrating how search space structure and inductive bias influence emergent behaviour.

<img width="640" height="670" alt="head-to-head-match-visualisation" src="https://github.com/user-attachments/assets/a7dc1987-01f2-4bea-9218-0453c3c4f7c7" />

---
