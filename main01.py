import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Tuple, Dict
import seaborn as sns
from scipy import stats

# Configuração para gráficos mais profissionais
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MarkovChain:
    """
    Classe para modelar e analisar Cadeias de Markov discretas com funcionalidades expandidas.

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

    def analyze_convergence_all_initial(self, max_steps: int = 50) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Analisa a convergência para a distribuição estacionária partindo de cada estado.

        Args:
            max_steps: Número máximo de passos a analisar

        Returns:
            Dicionário com convergência para cada estado inicial
        """
        pi = self.stationary_distribution()
        steps = np.arange(1, max_steps + 1)
        convergence_data = {}

        for i, initial_state in enumerate(self.states):
            # Distribuição inicial concentrada no estado i
            initial_dist = np.zeros(self.n_states)
            initial_dist[i] = 1.0
            
            norms = []
            for n in steps:
                P_n = self.n_step_transition(n)
                current_dist = initial_dist @ P_n
                norm = np.linalg.norm(current_dist - pi)
                norms.append(norm)
            
            convergence_data[initial_state] = (steps, np.array(norms))

        return convergence_data

    def sensitivity_analysis(self, perturbation: float = 0.1) -> Dict[str, Dict[str, float]]:
        """
        Realiza análise de sensibilidade da matriz de transição.

        Args:
            perturbation: Percentual de perturbação (ex: 0.1 = 10%)

        Returns:
            Dicionário com resultados da análise de sensibilidade
        """
        original_pi = self.stationary_distribution()
        original_return_times = self.mean_return_time()
        
        sensitivity_results = {}
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                param_name = f"P{i+1}{j+1}_{self.states[i]}->{self.states[j]}"
                
                # Perturbação positiva
                P_perturbed = self.P.copy()
                P_perturbed[i, j] *= (1 + perturbation)
                
                # Renormalizar linha para manter propriedade estocástica
                P_perturbed[i] = P_perturbed[i] / P_perturbed[i].sum()
                
                # Calcular nova distribuição estacionária
                mc_temp = MarkovChain(self.states, P_perturbed)
                new_pi = mc_temp.stationary_distribution()
                new_return_times = mc_temp.mean_return_time()
                
                # Calcular mudanças percentuais
                pi_changes = {}
                return_time_changes = {}
                
                for k, state in enumerate(self.states):
                    pi_change = ((new_pi[k] - original_pi[k]) / original_pi[k]) * 100
                    pi_changes[state] = pi_change
                    
                    rt_change = ((new_return_times[state] - original_return_times[state]) / 
                                original_return_times[state]) * 100
                    return_time_changes[state] = rt_change
                
                sensitivity_results[param_name] = {
                    'pi_changes': pi_changes,
                    'return_time_changes': return_time_changes,
                    'max_pi_change': max(abs(v) for v in pi_changes.values())
                }
        
        return sensitivity_results

    def monte_carlo_validation(self, steps: int = 10000, n_simulations: int = 100) -> Dict[str, Dict]:
        """
        Validação Monte Carlo partindo de diferentes estados iniciais.

        Args:
            steps: Número de passos por simulação
            n_simulations: Número de simulações por estado inicial

        Returns:
            Dicionário com resultados de validação
        """
        theoretical_pi = self.stationary_distribution()
        results = {}
        
        for initial_state in self.states:
            empirical_frequencies = {state: [] for state in self.states}
            
            for sim in range(n_simulations):
                path = self.simulate_path(initial_state, steps, random_seed=sim)
                unique, counts = np.unique(path, return_counts=True)
                freq_dict = dict(zip(unique, counts / len(path)))
                
                for state in self.states:
                    empirical_frequencies[state].append(freq_dict.get(state, 0))
            
            # Calcular estatísticas
            mean_frequencies = {state: np.mean(empirical_frequencies[state]) 
                              for state in self.states}
            std_frequencies = {state: np.std(empirical_frequencies[state]) 
                             for state in self.states}
            
            # Teste qui-quadrado
            observed = [len(empirical_frequencies[state]) * mean_frequencies[state] 
                       for state in self.states]
            expected = [len(empirical_frequencies[state]) * theoretical_pi[i] 
                       for i, state in enumerate(self.states)]
            
            chi2_stat, p_value = stats.chisquare(observed, expected)
            
            # RMS Error
            rms_error = np.sqrt(np.mean([(mean_frequencies[state] - theoretical_pi[i])**2 
                                       for i, state in enumerate(self.states)]))
            
            results[initial_state] = {
                'mean_frequencies': mean_frequencies,
                'std_frequencies': std_frequencies,
                'theoretical_pi': {state: theoretical_pi[i] for i, state in enumerate(self.states)},
                'chi2_stat': chi2_stat,
                'p_value': p_value,
                'rms_error': rms_error
            }
        
        return results

    def analyze_persistence(self, start_state: str, steps: int = 1000, n_simulations: int = 50) -> Dict:
        """
        Analisa a persistência dos estados através de simulações.

        Args:
            start_state: Estado inicial
            steps: Número de passos por simulação
            n_simulations: Número de simulações

        Returns:
            Dicionário com estatísticas de persistência
        """
        persistence_stats = {state: [] for state in self.states}
        
        for sim in range(n_simulations):
            path = self.simulate_path(start_state, steps, random_seed=sim)
            
            # Calcular sequências consecutivas para cada estado
            for state in self.states:
                consecutive_counts = []
                current_count = 0
                
                for day in path:
                    if day == state:
                        current_count += 1
                    else:
                        if current_count > 0:
                            consecutive_counts.append(current_count)
                        current_count = 0
                
                if current_count > 0:
                    consecutive_counts.append(current_count)
                
                if consecutive_counts:
                    persistence_stats[state].extend(consecutive_counts)
        
        # Calcular estatísticas
        results = {}
        for state in self.states:
            if persistence_stats[state]:
                results[state] = {
                    'mean_persistence': np.mean(persistence_stats[state]),
                    'max_persistence': np.max(persistence_stats[state]),
                    'std_persistence': np.std(persistence_stats[state]),
                    'median_persistence': np.median(persistence_stats[state])
                }
            else:
                results[state] = {
                    'mean_persistence': 0,
                    'max_persistence': 0,
                    'std_persistence': 0,
                    'median_persistence': 0
                }
        
        return results

    def plot_transition_graph(self, figsize: Tuple[int, int] = (12, 8),
                            min_edge_weight: float = 0.01):
        """
        Visualiza o grafo de transição da cadeia de Markov.
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
        node_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        plt.figure(figsize=figsize)

        # Desenha nós
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=4000, alpha=0.9)

        # Desenha arestas
        edges = G.edges(data=True)
        weights = [d['weight'] for (u, v, d) in edges]
        nx.draw_networkx_edges(G, pos, width=[w*6 for w in weights],
                              edge_color='gray', arrows=True, arrowsize=25,
                              connectionstyle="arc3,rad=0.15", alpha=0.7)

        # Labels
        nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', font_color='white')

        # Labels das arestas
        edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=12,
                                   font_color='red', font_weight='bold')

        plt.title("Grafo de Transição - Modelo Climático", fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_convergence_comparison(self, max_steps: int = 30):
        """
        Plota comparação de convergência para diferentes estados iniciais.
        """
        convergence_data = self.analyze_convergence_all_initial(max_steps)
        
        plt.figure(figsize=(12, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (initial_state, (steps, norms)) in enumerate(convergence_data.items()):
            plt.semilogy(steps, norms, color=colors[i], linewidth=3, 
                        marker='o', markersize=5, label=f'Inicial: {initial_state}',
                        alpha=0.8)
        
        plt.xlabel('Número de Passos', fontsize=14, fontweight='bold')
        plt.ylabel('||π^(n) - π||₂ (escala log)', fontsize=14, fontweight='bold')
        plt.title('Convergência para Distribuição Estacionária\n(Diferentes Condições Iniciais)', 
                 fontsize=16, fontweight='bold')
        plt.legend(fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_sensitivity_heatmap(self, perturbation: float = 0.1):
        """
        Plota heatmap da análise de sensibilidade.
        """
        sensitivity_results = self.sensitivity_analysis(perturbation)
        
        # Criar matriz para heatmap
        sensitivity_matrix = np.zeros((self.n_states, self.n_states))
        param_labels = []
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                param_name = f"P{i+1}{j+1}_{self.states[i]}->{self.states[j]}"
                sensitivity_matrix[i, j] = sensitivity_results[param_name]['max_pi_change']
                param_labels.append(f"{self.states[i]}→{self.states[j]}")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(sensitivity_matrix, 
                   xticklabels=self.states,
                   yticklabels=self.states,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlBu_r',
                   center=0,
                   cbar_kws={'label': 'Máxima Variação em π (%)'},
                   square=True)
        
        plt.title(f'Análise de Sensibilidade (±{perturbation*100:.0f}%)\nMáxima Variação na Distribuição Estacionária', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Estado Destino', fontsize=12, fontweight='bold')
        plt.ylabel('Estado Origem', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_monte_carlo_validation(self, steps: int = 10000, n_simulations: int = 100):
        """
        Plota validação Monte Carlo com barras de erro.
        """
        mc_results = self.monte_carlo_validation(steps, n_simulations)
        theoretical_pi = self.stationary_distribution()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (initial_state, results) in enumerate(mc_results.items()):
            ax = axes[idx]
            
            states_list = list(results['mean_frequencies'].keys())
            empirical_means = [results['mean_frequencies'][state] for state in states_list]
            empirical_stds = [results['std_frequencies'][state] for state in states_list]
            theoretical_values = [results['theoretical_pi'][state] for state in states_list]
            
            x = np.arange(len(states_list))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, empirical_means, width, 
                          yerr=empirical_stds, capsize=5,
                          label='Empírica (±1σ)', alpha=0.8, color=colors)
            bars2 = ax.bar(x + width/2, theoretical_values, width, 
                          label='Teórica', alpha=0.8, color='gray')
            
            ax.set_xlabel('Estados', fontweight='bold')
            ax.set_ylabel('Probabilidade', fontweight='bold')
            ax.set_title(f'Inicial: {initial_state}\nRMS Error: {results["rms_error"]:.4f}', 
                        fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(states_list)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Validação Monte Carlo ({n_simulations} simulações, {steps} passos)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_persistence_analysis(self, steps: int = 1000, n_simulations: int = 50):
        """
        Plota análise de persistência dos estados.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, initial_state in enumerate(self.states):
            persistence_stats = self.analyze_persistence(initial_state, steps, n_simulations)
            
            ax = axes[idx]
            states_list = list(persistence_stats.keys())
            mean_persistence = [persistence_stats[state]['mean_persistence'] for state in states_list]
            max_persistence = [persistence_stats[state]['max_persistence'] for state in states_list]
            
            x = np.arange(len(states_list))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, mean_persistence, width, 
                          label='Persistência Média', alpha=0.8, color=colors)
            bars2 = ax.bar(x + width/2, max_persistence, width, 
                          label='Persistência Máxima', alpha=0.6, color='gray')
            
            # Adicionar valores nas barras
            for i, (mean_val, max_val) in enumerate(zip(mean_persistence, max_persistence)):
                ax.text(i - width/2, mean_val + 0.1, f'{mean_val:.1f}', 
                       ha='center', va='bottom', fontweight='bold')
                ax.text(i + width/2, max_val + 0.1, f'{max_val}', 
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_xlabel('Estados', fontweight='bold')
            ax.set_ylabel('Dias Consecutivos', fontweight='bold')
            ax.set_title(f'Inicial: {initial_state}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(states_list)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Análise de Persistência ({n_simulations} simulações, {steps} passos)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def main():
    """Função principal expandida com todas as análises."""
    
    # Definição do problema: Modelo de Clima Simplificado
    states = ['Sol', 'Nublado', 'Chuva']
    
    # Matriz de transição (baseada no artigo)
    P = np.array([
        [0.7, 0.2, 0.1],  # Sol → {Sol: 70%, Nublado: 20%, Chuva: 10%}
        [0.3, 0.4, 0.3],  # Nublado → {Sol: 30%, Nublado: 40%, Chuva: 30%}
        [0.2, 0.3, 0.5]   # Chuva → {Sol: 20%, Nublado: 30%, Chuva: 50%}
    ])
    
    print("=" * 80)
    print("ANÁLISE COMPLETA DE CADEIA DE MARKOV - MODELO CLIMÁTICO EXPANDIDO")
    print("=" * 80)
    
    # Criar a cadeia de Markov
    mc = MarkovChain(states, P)
    
    # 1. Análise básica
    print("\n📊 MATRIZ DE TRANSIÇÃO:")
    print("-" * 40)
    for i, state in enumerate(states):
        row_str = " | ".join([f"{P[i,j]:.3f}" for j in range(len(states))])
        print(f"{state:>8} | {row_str}")
    
    # 2. Distribuição estacionária
    pi = mc.stationary_distribution()
    print("\n⚖️ DISTRIBUIÇÃO ESTACIONÁRIA:")
    print("-" * 40)
    for state, prob in zip(states, pi):
        print(f"{state:>8}: {prob:.4f} ({prob*100:.2f}%)")
    
    # 3. Tempos médios de retorno
    return_times = mc.mean_return_time()
    print("\n⏰ TEMPO MÉDIO DE RETORNO:")
    print("-" * 40)
    for state, time in return_times.items():
        print(f"{state:>8}: {time:.2f} dias")
    
    # 4. Análise de convergência para todos os estados iniciais
    print("\n📈 ANÁLISE DE CONVERGÊNCIA (TODOS OS ESTADOS INICIAIS):")
    print("-" * 60)
    convergence_data = mc.analyze_convergence_all_initial(20)
    
    for initial_state, (steps, norms) in convergence_data.items():
        convergence_step = np.where(norms < 1e-4)[0]
        if len(convergence_step) > 0:
            conv_step = convergence_step[0] + 1
            print(f"Inicial {initial_state:>8}: Convergência em {conv_step:2d} passos (norma: {norms[conv_step-1]:.2e})")
        else:
            print(f"Inicial {initial_state:>8}: Não convergiu em 20 passos")
    
    # 5. Análise de sensibilidade
    print("\n🔍 ANÁLISE DE SENSIBILIDADE (±10%):")
    print("-" * 50)
    sensitivity_results = mc.sensitivity_analysis(0.1)
    
    # Mostrar parâmetros mais sensíveis
    sorted_sensitivity = sorted(sensitivity_results.items(), 
                               key=lambda x: x[1]['max_pi_change'], reverse=True)
    
    print("Parâmetros mais sensíveis:")
    for param_name, results in sorted_sensitivity[:5]:
        param_short = param_name.split('_')[1]  # Extrair Sol->Nublado, etc.
        print(f"{param_short:>15}: Máx. variação {results['max_pi_change']:.2f}%")
    
    # 6. Validação Monte Carlo
    print("\n🎲 VALIDAÇÃO MONTE CARLO (1000 simulações, 10000 passos):")
    print("-" * 60)
    mc_results = mc.monte_carlo_validation(10000, 100)
    
    for initial_state, results in mc_results.items():
        print(f"\nInicial {initial_state}:")
        print(f"  RMS Error: {results['rms_error']:.4f}")
        print(f"  Chi² p-value: {results['p_value']:.4f}")
        for state in states:
            theoretical = results['theoretical_pi'][state]
            empirical = results['mean_frequencies'][state]
            std = results['std_frequencies'][state]
            print(f"  {state}: Teórico={theoretical:.4f}, Empírico={empirical:.4f}±{std:.4f}")
    
    # 7. Análise de persistência
    print("\n📊 ANÁLISE DE PERSISTÊNCIA:")
    print("-" * 40)
    for initial_state in states:
        persistence_stats = mc.analyze_persistence(initial_state, 1000, 50)
        print(f"\nInicial {initial_state}:")
        for state in states:
            stats = persistence_stats[state]
            if stats['mean_persistence'] > 0:
                print(f"  {state}: Média={stats['mean_persistence']:.1f} dias, "
                      f"Máx={stats['max_persistence']} dias")
    
    # Visualizações
    print("\n📈 Gerando visualizações expandidas...")
    
    # Grafo de transição aprimorado
    mc.plot_transition_graph()
    
    # Análise de convergência comparativa
    mc.plot_convergence_comparison()
    
    # Heatmap de sensibilidade
    mc.plot_sensitivity_heatmap()
    
    # Validação Monte Carlo com barras de erro
    mc.plot_monte_carlo_validation()
    
    # Análise de persistência
    mc.plot_persistence_analysis()
    
    print("\n✅ Análise completa finalizada!")
    print("\nEste código implementa todas as funcionalidades mencionadas no artigo:")
    print("- Análise de sensibilidade paramétrica")
    print("- Convergência com diferentes condições iniciais") 
    print("- Validação Monte Carlo expandida")
    print("- Análise de persistência dos estados")
    print("- Visualizações avançadas com intervalos de confiança")

if __name__ == "__main__":
    main()
