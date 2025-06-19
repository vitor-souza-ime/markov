import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple
import seaborn as sns

# Configuração para gráficos mais profissionais
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MarkovChain:
    """
    Classe para modelar e analisar Cadeias de Markov discretas.

    Attributes:
        states (List[str]): Lista dos estados possíveis
        P (np.ndarray): Matriz de transição
        n_states (int): Número de estados
    """

    def __init__(self, states: List[str], transition_matrix: np.ndarray):
        """
        Inicializa a Cadeia de Markov.

        Args:
            states: Lista com nomes dos estados
            transition_matrix: Matriz de transição (n×n)

        Raises:
            AssertionError: Se a matriz não for estocástica
        """
        self.states = states
        self.P = np.array(transition_matrix)
        self.n_states = len(states)

        # Validações
        self._validate_transition_matrix()

    def _validate_transition_matrix(self):
        """Valida se a matriz de transição é estocástica."""
        assert self.P.shape == (self.n_states, self.n_states), \
            "Dimensões da matriz incompatíveis com número de estados"
        assert np.allclose(self.P.sum(axis=1), 1, rtol=1e-10), \
            "A matriz de transição deve ser estocástica (linhas somam 1)"
        assert np.all(self.P >= 0), \
            "Probabilidades devem ser não-negativas"

    def simulate_path(self, start_state: str, steps: int = 100,
                     random_seed: int = None) -> List[str]:
        """
        Simula uma trajetória da cadeia de Markov.

        Args:
            start_state: Estado inicial
            steps: Número de passos a simular
            random_seed: Semente para reprodutibilidade

        Returns:
            Lista com a sequência de estados visitados
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        current = self.states.index(start_state)
        path = [start_state]

        for _ in range(steps):
            current = np.random.choice(self.n_states, p=self.P[current])
            path.append(self.states[current])

        return path

    def stationary_distribution(self) -> np.ndarray:
        """
        Calcula a distribuição estacionária π tal que πP = π.

        Returns:
            Vetor com a distribuição estacionária
        """
        # Método dos autovalores/autovetores
        eigenvals, eigenvecs = np.linalg.eig(self.P.T)

        # Encontra autovalor mais próximo de 1
        idx = np.argmin(np.abs(eigenvals - 1.0))
        pi = eigenvecs[:, idx].real

        # Normaliza para garantir que soma 1
        pi = np.abs(pi) / np.sum(np.abs(pi))

        return pi

    def n_step_transition(self, n: int) -> np.ndarray:
        """
        Calcula a matriz de transição em n passos: P^n

        Args:
            n: Número de passos

        Returns:
            Matriz P^n
        """
        return np.linalg.matrix_power(self.P, n)

    def mean_return_time(self) -> dict:
        """
        Calcula o tempo médio de retorno para cada estado.

        Returns:
            Dicionário com tempos médios de retorno
        """
        pi = self.stationary_distribution()
        return {state: 1/prob for state, prob in zip(self.states, pi)}

    def analyze_convergence(self, max_steps: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analisa a convergência para a distribuição estacionária.

        Args:
            max_steps: Número máximo de passos a analisar

        Returns:
            Tupla com (passos, normas da diferença)
        """
        pi = self.stationary_distribution()
        steps = np.arange(1, max_steps + 1)
        norms = []

        initial_dist = np.array([1, 0, 0])  # Começando do primeiro estado

        for n in steps:
            P_n = self.n_step_transition(n)
            current_dist = initial_dist @ P_n
            norm = np.linalg.norm(current_dist - pi)
            norms.append(norm)

        return steps, np.array(norms)

    def plot_transition_graph(self, figsize: Tuple[int, int] = (10, 8),
                            min_edge_weight: float = 0.01):
        """
        Visualiza o grafo de transição da cadeia de Markov.

        Args:
            figsize: Dimensões da figura
            min_edge_weight: Peso mínimo para mostrar aresta
        """
        G = nx.DiGraph()

        # Adiciona nós e arestas
        for i in range(self.n_states):
            G.add_node(self.states[i])
            for j in range(self.n_states):
                if self.P[i, j] > min_edge_weight:
                    G.add_edge(self.states[i], self.states[j],
                             weight=self.P[i, j])

        # Layout e cores
        pos = nx.circular_layout(G)
        node_colors = plt.cm.Set3(np.linspace(0, 1, self.n_states))

        plt.figure(figsize=figsize)

        # Desenha nós
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=3000, alpha=0.8)

        # Desenha arestas com espessura proporcional ao peso
        edges = G.edges(data=True)
        weights = [d['weight'] for (u, v, d) in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                              edge_color='gray', arrows=True, arrowsize=20,
                              connectionstyle="arc3,rad=0.1")

        # Labels dos nós
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

        # Labels das arestas
        edge_labels = {(u, v): f"{d['weight']:.2f}"
                      for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10,
                                   font_color='red')

        plt.title("Grafo de Transição da Cadeia de Markov",
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_convergence_analysis(self, max_steps: int = 50):
        """
        Plota a análise de convergência para distribuição estacionária.

        Args:
            max_steps: Número máximo de passos
        """
        steps, norms = self.analyze_convergence(max_steps)

        plt.figure(figsize=(10, 6))
        plt.semilogy(steps, norms, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Número de Passos', fontsize=12)
        plt.ylabel('||π^(n) - π||₂ (escala log)', fontsize=12)
        plt.title('Convergência para Distribuição Estacionária',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_simulation_analysis(self, start_state: str, steps: int = 1000,
                               random_seed: int = 42):
        """
        Plota análise da simulação comparando com teoria.

        Args:
            start_state: Estado inicial
            steps: Número de passos
            random_seed: Semente aleatória
        """
        # Simula trajetória
        path = self.simulate_path(start_state, steps, random_seed)

        # Calcula frequências empíricas
        unique, counts = np.unique(path, return_counts=True)
        empirical_freq = dict(zip(unique, counts / len(path)))

        # Distribuição teórica
        pi = self.stationary_distribution()
        theoretical_freq = {state: prob for state, prob in zip(self.states, pi)}

        # Plot comparativo
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Frequências
        states_plot = self.states
        empirical_values = [empirical_freq.get(s, 0) for s in states_plot]
        theoretical_values = [theoretical_freq[s] for s in states_plot]

        x = np.arange(len(states_plot))
        width = 0.35

        ax1.bar(x - width/2, empirical_values, width, label='Empírica', alpha=0.8)
        ax1.bar(x + width/2, theoretical_values, width, label='Teórica', alpha=0.8)
        ax1.set_xlabel('Estados')
        ax1.set_ylabel('Frequência')
        ax1.set_title('Frequências: Empírica vs Teórica')
        ax1.set_xticks(x)
        ax1.set_xticklabels(states_plot)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Evolução temporal das frequências
        window_size = min(100, steps // 10)
        time_points = range(window_size, len(path), window_size)

        for i, state in enumerate(self.states):
            frequencies = []
            for t in time_points:
                window = path[t-window_size:t]
                freq = window.count(state) / len(window)
                frequencies.append(freq)

            ax2.plot(time_points, frequencies, label=f'{state}',
                    marker='o', markersize=3, linewidth=2)
            ax2.axhline(y=pi[i], color=f'C{i}', linestyle='--', alpha=0.7)

        ax2.set_xlabel('Passos')
        ax2.set_ylabel('Frequência')
        ax2.set_title('Convergência das Frequências Empíricas')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return empirical_freq, theoretical_freq

def main():
    """Função principal para demonstrar o uso da classe."""

    # Definição do problema: Modelo de Clima Simplificado
    states = ['Sol', 'Nublado', 'Chuva']

    # Matriz de transição (interpretação meteorológica realística)
    P = np.array([
        [0.7, 0.2, 0.1],  # Sol → {Sol: 70%, Nublado: 20%, Chuva: 10%}
        [0.3, 0.4, 0.3],  # Nublado → {Sol: 30%, Nublado: 40%, Chuva: 30%}
        [0.2, 0.3, 0.5]   # Chuva → {Sol: 20%, Nublado: 30%, Chuva: 50%}
    ])

    print("=" * 60)
    print("ANÁLISE DE CADEIA DE MARKOV - MODELO CLIMÁTICO")
    print("=" * 60)

    # Cria a cadeia de Markov
    mc = MarkovChain(states, P)

    # 1. Análise da matriz de transição
    print("\n📊 MATRIZ DE TRANSIÇÃO:")
    print("-" * 30)
    for i, state in enumerate(states):
        row_str = " | ".join([f"{P[i,j]:.3f}" for j in range(len(states))])
        print(f"{state:>8} | {row_str}")

    # 2. Distribuição estacionária
    pi = mc.stationary_distribution()
    print("\n⚖️ DISTRIBUIÇÃO ESTACIONÁRIA:")
    print("-" * 35)
    for state, prob in zip(states, pi):
        print(f"{state:>8}: {prob:.4f} ({prob*100:.2f}%)")

    # 3. Tempos médios de retorno
    return_times = mc.mean_return_time()
    print("\n⏰ TEMPO MÉDIO DE RETORNO:")
    print("-" * 30)
    for state, time in return_times.items():
        print(f"{state:>8}: {time:.2f} dias")

    # 4. Simulação
    print("\n🎲 SIMULAÇÃO (15 dias, começando com Sol):")
    print("-" * 45)
    trajectory = mc.simulate_path('Sol', 15, random_seed=42)
    traj_str = ' → '.join(trajectory)
    print(f"Trajetória: {traj_str}")

    # 5. Análise de convergência em n passos
    print("\n📈 PROBABILIDADES EM N PASSOS (partindo de Sol):")
    print("-" * 50)
    for n in [1, 5, 10, 20]:
        P_n = mc.n_step_transition(n)
        probs = P_n[0]  # Primeira linha (partindo de Sol)
        print(f"n={n:2d}: ", end="")
        for i, state in enumerate(states):
            print(f"{state}: {probs[i]:.4f} ", end="")
        print()

    # Visualizações
    print("\n📈 Gerando visualizações...")

    # Grafo de transição
    mc.plot_transition_graph()

    # Análise de convergência
    mc.plot_convergence_analysis(30)

    # Análise de simulação
    emp_freq, theo_freq = mc.plot_simulation_analysis('Sol', 5000)

    print("\n✅ Análise completa!")

if __name__ == "__main__":
    main()
