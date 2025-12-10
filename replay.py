import json
import numpy as np # type: ignore

from classes import BattleManager

# Observa em tempo real o genoma salvo jogando o RPG
# Claro, faça o treinamento do genoma anterior
# No repo estará salvo o output de um treinamento prévio

# Carrega o genoma
def load_champion(filename="champion.json"):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
        return np.array(data)
    except FileNotFoundError:
        print("Erro: Arquivo 'champion.json' não encontrado.")
        print("Rode o 'train.py' primeiro para gerar um genoma.")
        exit()

# Carrega o sistema RPG e assiste o genoma jogando
def watch_mode():
    # 1. Carrega o genoma
    champion_genome = load_champion()
    print("Genoma Carregado")
    
    # 2. Prepara a Arena
    battle = BattleManager(verbose=True) # Ativa prints detalhados
    battle.active_genome = champion_genome # type: ignore
    
    while True:
        print("  NOVA BATALHA DE EXIBIÇÃO  ")
        
        battle.cleanup()
        # Roda a luta
        outcome, dmg, _, _ = battle.battleLoop()
        res = "VITÓRIA" if outcome == 1 else "DERROTA" if outcome == -1 else "EMPATE"
        print(f"\n>>> RESULTADO: {res} | Dano Causado: {dmg}")
        
        # Se o usuário quer ver mais (ad infinitum) ou acabar a execução
        user_input = input("\n[Enter] para próxima luta, [S] para sair: ").lower()
        if user_input == 's':
            break

if __name__ == "__main__":
    watch_mode()