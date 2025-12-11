import json
import random
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

from classes import *

# Código de Treinamento do modelo
# Treina do zero por X Gerações com população Y cada
# Aplica uma mutação com intensidade regressiva temporalmente
# Exporta uma imagem do gráfico do treinamento e salva o melhor genoma encontrado

# Parâmetros de Treino
POPULATION_SIZE = 100 # Quantos genomas terão por geração 
GENERATIONS = 10000 # Quantas gerações terão no treino
NUM_TESTS = 20 # Quantas rodadas um genoma deve enfrentar
VERBOSE = False # Veja o combate acontecendo para o genoma sendo treinado
                # Mas use parâmetros de treino bem baixos (ou vai demorar 50 anos pra acabar)
DELTA_P = 25.0 # Diferença mínima de fitness exigida para não considerar estagnado
BASE_MUTATION_RATE = 0.1 # Taxa base de mutação
BASE_MUTATION_SIGMA = 0.2 # Sigma base de mutação
FRACTION = 2 # Qual fração 1/FRACTION dos top melhores genomas será passado adiante

# Lê o moveList
with open('movelist.json', 'r') as f:
    moveSheet = json.load(f)
f.close()

# Salva o melhor output encontrado do treinamento
# O genoma pode ser utilizado no replay.py para ver ele em ação
def save_genome(genome, filename="champion.json"):
    data = genome.tolist()
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Genoma salvo em: {filename}")

# A eficiência de um genoma de acordo com as métricas de batalha
# Maior = Melhor
def calculate_fitness(outcome, turns, hero_hp_pct, damage_dealt_total, enemy_max_hp_total, hero_died_with_resources, effective_heal_total):
    fitness = 0.0
    damage_score = (damage_dealt_total / max(100, enemy_max_hp_total)) * 200
    fitness += damage_score * 1
    
    fitness += effective_heal_total * 2
    
    if outcome == 1: # Vitória
        fitness += 800 
        fitness += (50 - turns) * 20 
        fitness += hero_hp_pct * 100
    elif outcome == -1: # Derrota
        if hero_died_with_resources:
            fitness -= 800 # Não sabe se curar, poderia ter se curado, inutil
        else:
            fitness -= 400 # Punição normal por perder
    else: # Empate
        fitness -= 300
    return max(0, fitness)


def mutation(counter, gen):
    mutation_status = [
        (1, "Normal Rate x1"),
        (0.5,"Fine Tuning x0.5"),
        (0.25,"Fine Tuning x0.25"),
        (2,"Mutation Increase x2"),
        (4,"Mutation Increase x4"),
        (8,"Mutation Increase x8"),
        (16,"Mutation Increase x16"),
        (-1,"Genocide"),
    ]
    status = mutation_status[int(counter / 10)]
    if counter % 10 == 0 and gen % 10 == 0: print(status[1])
    return status[0]

# Função principal de treino
# Exporta o melhor genoma e o gráfico de desempenho como arquivos
def train():
    
    battle = BattleManager(verbose=VERBOSE) # Ativa o sistema de batalha
    dummy = AIBrain() # Falso cérebro para extrair o tamanho do genoma, não é utilizado mais depois
    genome_size = len(dummy.genome)
    
    # Tensor de população e genoma de cada
    population = [np.random.uniform(-1, 1, genome_size) for _ in range(POPULATION_SIZE)]
    
    # Histórico para o gráfico
    history_max_fitness = []
    history_avg_fitness = []
    history_winrate = []
    history_multiplier = []
    
    # Mutação Adaptativa
    fitness_buffer = [] # Buffer das médias das últimas gerações
    stagnation_counter = 0 # O contador mestre (10, 20, 30...)

    current_mutation_rate = BASE_MUTATION_RATE
    current_mutation_sigma = BASE_MUTATION_SIGMA
    
    print(f"Parâmetros de Treino:\n\tGerações: {GENERATIONS}\n\tPopulação: {POPULATION_SIZE}\n\tRodadas: {NUM_TESTS}")
    
    # Loop geracional
    for gen in range(GENERATIONS):
        # Parâmetros de mutação geracional
         
        generation_fitness = []
        tally = [0,0,0] # [Derrota, Empate, Vitoria]
        
        # Loop populacional
        for i, genome in enumerate(population):
            battle.active_genome = genome  # type: ignore
            total_fit = 0
            
            # Loop de batalhas
            for _ in range(NUM_TESTS):
                battle.cleanup() 
                outcome, dmg, foeMaxHP, totalHealed = battle.battleLoop()
                tally[outcome + 1] += 1
                
                hero = battle.charList[0] 
                died_dumb = False # Não se curou e morreu por causa disto
                if outcome == -1: # Se morreu
                    # Verifica se tinha algum move de cura (Target=1, Type=2) 
                    # E se tinha SP/MP pra usar tal move
                    has_heal_move = False
                    for m_id in hero.moveList:
                        m_data = moveSheet[m_id] # Precisaria importar moveSheet no train.py ou passar
                        if m_data['Target'] == 1 and m_data['Type'] == 2: # É cura
                             if hero.curSP >= m_data['SPCost'] and hero.curMP >= m_data['MPCost']:
                                 has_heal_move = True
                                 break
                    if has_heal_move: # Se tinha, marca o genoma como "idiota"
                        died_dumb = True
                
                hero_hp_pct = hero.curHP / hero.stats['HP'] # % de vida restante
                turns = battle.turn # Total de turnos
                
                # Obtém o fitness da batalha
                fit = calculate_fitness(outcome, turns, hero_hp_pct, dmg, foeMaxHP, died_dumb, totalHealed)
                total_fit += fit
            
            avg_fitness = total_fit / NUM_TESTS
            generation_fitness.append(avg_fitness)
            
        # Métricas da Geração
        winrate = tally[2] / sum(tally)
        best_gen_fit = np.max(generation_fitness)
        avg_gen_fit = np.mean(generation_fitness)
        
        history_winrate.append(winrate)
        history_max_fitness.append(best_gen_fit)
        history_avg_fitness.append(avg_gen_fit)
        
        print(f"Gen {gen}: Winrate {winrate*100:.1f}% | MaxFit {best_gen_fit:.0f} | AvgFit {avg_gen_fit:.0f}")

        # Estagnação:
        fitness_buffer.append(avg_gen_fit) # Adiciona no buffer
        if len(fitness_buffer) >= 20:
            # Pega os ultimos 20 valores médios
            window = fitness_buffer[-20:]
            # Média dos 10 primeiras vs 10 últimas desta janela
            avg_old = sum(window[:10]) / 10
            avg_new = sum(window[10:]) / 10
            
            delta = avg_new - avg_old # Quantos aumentou
            
            if delta < DELTA_P:
                # Estagnou, aumenta o counter
                stagnation_counter += 1
                print(f"Estagnou, counter: {stagnation_counter}")
            else:
                # Melhorou! Zera o contador (ou decrementa se quiser ser bonzinho)
                stagnation_counter = 0
                
            # Limpa o buffer
            fitness_buffer = []
        
        # Multiplicador de Mutação:
        mutation_multiplier = mutation(stagnation_counter, gen)
        history_multiplier.append(mutation_multiplier)
        if mutation_multiplier != -1: # Mutação Normal
            current_mutation_rate = BASE_MUTATION_RATE * mutation_multiplier
            current_mutation_sigma = BASE_MUTATION_SIGMA * mutation_multiplier
        else: # Genocídio
            # Pega o último melhor genoma
            stagnation_counter = 0 # Reseta o contator
            fitness_buffer = [] # Esvazia o buffer
            current_mutation_rate = BASE_MUTATION_RATE # Retorna ao normal
            current_mutation_sigma = BASE_MUTATION_RATE
            
        # Seleção:
        new_pop = []
        best_idx = np.argmax(generation_fitness)
        champion = population[best_idx] # Melhor genoma geracional
        
        if mutation_multiplier == -1: # Genocídio
            new_pop.append(champion.copy())
            while len(new_pop) < POPULATION_SIZE:
                 random_child = np.random.uniform(-1, 1, genome_size)
                 new_pop.append(random_child)
        else: # Reprodução Normal
            # Elitismo: Top 1 serão preservados
            sorted_indices = np.argsort(generation_fitness)[::-1]
            for i in range(3):
                new_pop.append(population[sorted_indices[i]].copy())

            # Os melhores da fração 1/FRACTION serão guardados
            # Ex: FRACTION = 2, .°. top 50% será passado adiante e o resto cortado fora
            top_half_indices = sorted_indices[:int(POPULATION_SIZE/FRACTION)]

            # Reprodução
            while len(new_pop) < POPULATION_SIZE:
                # Crossbreeding assexual
                idx_p1 = np.random.choice(top_half_indices)
                idx_p2 = np.random.choice(top_half_indices)

                parent1 = population[idx_p1]
                parent2 = population[idx_p2]

                child = parent1.copy()
                cut = random.randint(0, genome_size-1)
                child[cut:] = parent2[cut:] # Quanto será passado adiante do segundo parente

                # Mutação
                if random.random() < current_mutation_rate:
                    mutation_points = random.randint(1, 3) # Quantos casos de mutação ocorrerão
                    for _ in range(mutation_points):
                        idx = random.randint(0, genome_size-1) # Alguma seção aleatória do genoma
                        child[idx] += np.random.normal(0, current_mutation_sigma) # Adiciona uma alteração aleatória baseada na curva gaussiana

                new_pop.append(child)

        population = new_pop
                
    print("Treino Finalizado!")
    
   # Gráfico
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(history_max_fitness, label='Melhor Fitness', color='green')
    plt.plot(history_avg_fitness, label='Fitness Médio', color='blue')
    plt.title('Evolução do Fitness')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(history_winrate, label='Taxa de Vitória', color='orange')
    plt.title('Taxa de Vitória')
    plt.ylim(0, 1.0)
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(history_multiplier, label='Multiplicador', color='purple')
    plt.title('Taxa de Mutação')
    plt.ylim(-2, 17)
    plt.grid(True)
    
    filename_img = 'resultado_treino.png'
    plt.savefig(filename_img)
    plt.close()
    print(f"Gráfico salvo como: {filename_img}")
    
    # Genoma Campeão
    save_genome(champion, "champion.json")

if __name__ == "__main__":
    train()