````markdown
# Markov Weather Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](https://opensource.org/licenses/MIT)

This project implements a **Discrete-Time Markov Chain** to simulate and analyze weather behavior. The model uses a simple transition matrix between three weather states: **Sunny**, **Cloudy**, and **Rainy**. It includes simulation, convergence analysis, mean return times, and graphical visualizations.

📁 **Repository:** [https://github.com/vitor-souza-ime/markov](https://github.com/vitor-souza-ime/markov)

## 🧠 Features

- Custom Markov chain implementation with any state set and transition matrix
- Validation of stochastic properties
- Simulation of state trajectories
- Computation of stationary distribution
- Mean return time for each state
- n-step transition probability analysis
- Graphical visualizations using:
  - NetworkX (transition graph)
  - Matplotlib & Seaborn (convergence, frequency comparisons)

## 📦 Requirements

- Python 3.8+
- NumPy 1.21.0
- Matplotlib 3.4.2
- NetworkX 2.6.2
- Seaborn 0.11.1

Install the dependencies with:

```bash
pip install numpy==1.21.0 matplotlib==3.4.2 networkx==2.6.2 seaborn==0.11.1
````

## ▶️ Usage

Run the main script:

```bash
python main.py
```

Sample output includes:

* Transition matrix
* Stationary distribution
* Mean return times
* Simulated trajectory
* Probability evolution over time
* Plots and diagrams

## 📚 Example

The model uses the following transition matrix:

```
           Sunny   Cloudy   Rainy
Sunny     [0.7,    0.2,     0.1]
Cloudy    [0.3,    0.4,     0.3]
Rainy     [0.2,    0.3,     0.5]
```

Which means, for example, if today is sunny, there's a:

* 70% chance tomorrow is sunny
* 20% chance of cloudy
* 10% chance of rain

## 📊 Visual Outputs

The script generates:

* A directed graph of state transitions
* Convergence plot to the stationary distribution
* Comparison between empirical and theoretical frequencies

## 📁 File Structure

```
.
├── main.py        # Main script with MarkovChain class and execution
├── README.md      # Project documentation
```

## 🔬 References

* HARRIS et al. (2020) — NumPy
* HUNTER (2007) — Matplotlib
* HAGBERG et al. (2008) — NetworkX
* WASKOM (2021) — Seaborn

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/licenses/MIT) file for details.

---
