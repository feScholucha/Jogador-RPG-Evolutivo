import numpy as np # type: ignore
import json
import random
import time

# O núcleo do código
# Está aqui o código do sistema RPG, o algoritmo procedural, e o cérebro evolutivo

# Parâmetros
PLAYER_OVERRIDE = False # Desativa o sistema automático
                        # E ativa o algoritmo procedural no lugar
                        # Deveria ativar um sistema de controle do usuário
                        # Mas não foi implementado nesta versão

# Abre o arquivo de movimentos
with open('movelist.json', 'r') as f:
    moveSheet = json.load(f)

# Abre o arquivo de personagens
with open('charSheet.json', 'r') as f:
    charSheet = json.load(f)

# Cheat Sheet: (O que cada coisa significa)
# Function: Ataque ou Buff
# 0 : Attack
# 1 : Buff
#
# Type: Qual o Tipo usado
# 0: Physical
# 1: Ranged
# 2: Magic
#
# Target: Quem está sendo atingido
# 0: Single
# 1: Self
# 2: Multi
#
# Element: Qual elemento é o ataque
# 0 : Normal
# 1 : Fire
# 2 : Water
# 3 : Plant
# 4 : Light
# 5 : Dark 

# Matchup Elementais, (ex: Fogo é fraco contra Água)
TYPE_CHART = {
    (1, 3): 2.0,
    (1, 2): 0.5,
    (1, 1): 0.5,
    (2, 1): 2.0,
    (2, 3): 0.5,
    (2, 2): 0.5,
    (3, 2): 2.0,
    (3, 1): 0.5,
    (3, 3): 0.5,
    (4, 5): 2.0,
    (4, 4): 0.5,
    (5, 4): 2.0,
    (5, 5): 0.5,
}

# O cérebro responsável por fazer o genoma servir para alguma coisa
class AIBrain:
    def __init__(self, genome=None, input_size=12, hidden_size=8):
        # A arquitetura utilizada de 3 camadas, Input -> Hidden -> Output
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1 # O score de quão boa é essa ação escolhida pelo genoma
        
        # Se não há um genoma, cria um novo
        # Utilizado pela primeira geração
        if genome is None:
            # Pesos da camada 1 + Bias 1 + Pesos da camada 2 + Bias 2
            n_weights = (input_size * hidden_size) + hidden_size + (hidden_size * 1) + 1
            self.genome = np.random.uniform(-1, 1, n_weights)
        else:
            self.genome = np.array(genome)
            
    def predict(self, inputs):
        # Desempacota o genoma linear em pesos de matriz
        idx1 = self.input_size * self.hidden_size
        W1 = self.genome[0:idx1].reshape((self.input_size, self.hidden_size))
        
        idx2 = idx1 + self.hidden_size
        B1 = self.genome[idx1:idx2]
        
        idx3 = idx2 + (self.hidden_size * 1)
        W2 = self.genome[idx2:idx3].reshape((self.hidden_size, 1))
        
        B2 = self.genome[idx3:]
        
        # Forward Pass
        # Camada Oculta com ativação ReLU
        z1 = np.dot(inputs, W1) + B1
        a1 = np.maximum(0, z1) 
        
        # Saída Linear (pode ser negativa ou positiva)
        output = np.dot(a1, W2) + B2
        return output[0] # Retorna o float do score

# Sistema estático responsável por fornecer qual ataque deve ser usado e e em quem
class CombatAlgorithms:
    # Retorna o melhor move e os alvos que pode ser utilizado, de acordo com o algoritmo utilizado
    @classmethod
    def getMove(cls, ID, isHero, situation, moveList, current_genome=None):
        # Se é um inimigo, usa o algoritmo procedural
        if isHero == False:
            return cls.dumbProceduralAttack(ID, isHero, situation, moveList)
        # Se é o herói e tem um genoma, usa o sistema evolutivo
        if PLAYER_OVERRIDE == False and current_genome is not None:
            # Ativa o cerebro
            brain = AIBrain(genome=current_genome)
            
            best_score = -99999
            best_move = moveList[0]
            best_targets = []
            
            # Obter os status do héroi (me)
            me = next(x for x in situation if x['battleID'] == ID)
            
            # Iterar sobre cada golpe possível
            for move_id in moveList:
                move_data = moveSheet[move_id] # Puxando do JSON global
                possible_targets = []
                
                # Definir quem são os alvos válidos para este golpe
                if move_data['Target'] == 1: # Self
                    possible_targets = [me]
                elif move_data['Target'] == 2: # AoE (Todos inimigos)
                    # No AoE, considera o "alvo principal" como o primeiro inimigo vivo só pra gerar input
                    enemies = [x for x in situation if x['isHero'] != isHero and x['isAlive']]
                    if not enemies: continue
                    possible_targets = [enemies[0]] # Simplificação: Avalia o AoE baseado no primeiro inimigo
                else: # Single Target (Inimigos)
                    possible_targets = [x for x in situation if x['isHero'] != isHero and x['isAlive']]
                
                # O cérebro avalia cada par (Golpe, Alvo)
                for target in possible_targets:
                    # Gera os inputs
                    inputs = cls.get_action_inputs(me, target, move_data, situation)
                    
                    # Cérebro dá a nota
                    score = brain.predict(inputs)

                    ## OBS: Injeção procedural de teste, não é uma escolha evolutiva
                    # if inputs[0] > 0.7 and inputs[6] == 1.0: # Se está quase morto e decidiu curar
                    #     score += 2.0 # Boost de cura crítica, feedback positivo pela escolha
                        
                    if score > best_score:
                        best_score = score
                        best_move = move_id
                        
                        # Definir a lista final de alvos baseada no tipo
                        if move_data['Target'] == 2:
                             # Se escolheu AoE, pega todos IDs de inimigos vivos
                             all_enemies = [x['battleID'] for x in situation if x['isHero'] != isHero and x['isAlive']]
                             best_targets = all_enemies
                        else:
                             best_targets = [target['battleID']]
            
            return best_move, best_targets
        
        # Fallback se não tiver genoma ou for player manual (não implementado ainda)
        else:    
            return cls.dumbProceduralAttack(ID, isHero, situation, moveList)
        
    # Retorna o multiplicador do matchup elemental
    @staticmethod
    def getMultiplier(atkElem, defElem):
        return TYPE_CHART.get((atkElem, defElem), 1.0)

    # Algoritmo procedural que pega o alvo com menos vida e usa o golpe disponível mais forte nele
    @staticmethod
    def dumbProceduralAttack(ID, isHero, situation, moveList):
        strongestMove = 0
        bestPower = -1
        # Ignora matchups e acha "o maior porrete", mesmo que seja pior que outra opção
        for x in moveList:
            if moveSheet[x]["BasePower"] > bestPower and moveSheet[x]["Target"] != 1:
                strongestMove = x
                bestPower = moveSheet[x]["BasePower"]
        
        # Se o melhor ataque é um MultiTarget
        if moveSheet[strongestMove]["Target"] == 2:
            enemyList = []
            for x in situation:
                if x["isHero"] != isHero and x["isAlive"]:
                    enemyList.append(x["battleID"])
            # Não precisa decidir se vai todo mundo apanhar
            return strongestMove, enemyList

        # Pega o inimigo com a menor vida (absoluta, não relativa)
        weakestEnemy = -1
        lowestHP = 9999999
        for x in situation:
            if x["isHero"] != isHero and x["isAlive"]:
                if x["HP"] < lowestHP:
                    weakestEnemy = x["battleID"]
                    lowestHP = x["HP"]
        return strongestMove, [weakestEnemy]
  
    # Retorna a situação de combate atual
    @staticmethod
    def get_action_inputs(attacker_stats, target_stats, move_data, situation):
        
        # Normalizações de HP e SP (Entidade)       
        raw_hp = attacker_stats['HP'] / attacker_stats['MaxHP']
        my_hurt_level = min(1.0, 1.0 - raw_hp)
        my_sp = min(1.0, attacker_stats['SP'] / attacker_stats['MaxSP'])
        
        # Normalizações de HP (Alvo)
        tgt_hp = min(1.0, target_stats['HP'] / target_stats['MaxHP']) if target_stats else 0
        is_self = 1.0 if attacker_stats['battleID'] == target_stats['battleID'] else 0.0
        
        # Vida Absoluta (Alvo)
        tgt_max_hp_raw = target_stats['MaxHP'] if target_stats else 0
        tgt_bulk = min(1.0, tgt_max_hp_raw / 1000.0)
        
        # Informações do movimento
        cost = move_data['SPCost'] / 50.0 # Normalizando custo
        power = move_data['BasePower'] / 100.0
        is_heal = 1.0 if move_data['Target'] == 1 and move_data['Type'] == 2 else 0.0
        is_aoe = 1.0 if move_data['Target'] == 2 else 0.0
        
        # Vantagem Elemental
        atk_elem = move_data['Element']
        def_elem = target_stats['Element'] if target_stats else 0
        mult = CombatAlgorithms.getMultiplier(atk_elem, def_elem)
        advantage = 0.0
        if mult > 1.0: advantage = 1.0
        elif mult < 1.0: advantage = -1.0
        
        # Número de Inimigos
        enemy_count = 0
        for entity in situation:
            if not entity['isHero'] and entity['isAlive']:
                enemy_count += 1
        enemy_density = enemy_count / 3.0 # Normalizado
        chaos = len(situation) / 4.0
        
        # Monta o vetor de 10 inputs
        inputs = np.array([
            my_hurt_level,          # Preciso de cura?
            my_sp,                  # Tenho energia?
            tgt_hp,                 # Inimigo está quase morrendo?
            tgt_bulk,               # Inimigo tem mta vida absoluta?
            is_self,                # É em mim?
            cost,                   # É caro?
            power,                  # É forte?
            is_heal,                # É cura?
            is_aoe,                 # É área?
            advantage,              # É efetivo?
            chaos,                  # Cáos total
            enemy_density           # Quantos Inimigos?
        ])
        
        return inputs
    
# Estrutura de personagem, utilizada para cada ator ativo no combate
class Character:
    def __init__(self, name, typeID, ID, hero, statSheet = None, moveList = [], genome=None):
        self.ID = ID
        self.name = name
        self.typeID = typeID
        self.isHero = hero
        self.isAlive = True
        self.stats = self._setBaseStats(statSheet)
        self.curHP = self.stats["HP"]
        self.curSP = 0.5 * self.stats["SP"]
        self.curMP = self.stats["MP"]
        self.moveList = moveList
        self.genome = genome
        
    def _setBaseStats(self, statSheet = None):
        if statSheet == None:
            exStatSheet = {
                "HP" : 100,
                "BaseElement" : 0,
                "SP" : 20,
                "MP" : 20,
                "Str" : 10,
                "Dex" : 10,
                "Int" : 10,
                "Def" : 10,
                "Wis" : 10,
            }
            return exStatSheet
        return statSheet
    
    def setHP(self, value):
        self.curHP = value
        if self.curHP < 0:
            self.curHP = 0
        if self.curHP > 0:
            self.isAlive = True
        if self.curHP > self.stats["HP"]:
            self.curHP = self.stats["HP"]
        
    def addHP(self, value):
        preHP = self.curHP
        if self.curHP > 0:
            self.curHP += value
        if self.curHP <= 0:
            self.isAlive = False
            self.curHP = 0
        if self.curHP > self.stats["HP"]:
            self.curHP = self.stats["HP"]
        postHP = self.curHP
        return postHP-preHP
            
    def setMP(self, value):
        self.curMP = value
        if self.curMP < 0:
            self.curMP = 0
        if self.curMP > 0:
            self.isAlive = True
        if self.curMP > self.stats["MP"]:
            self.curMP = self.stats["MP"]
        
    def addMP(self, value):
        if self.curMP > 0:
            self.curMP += value
        if self.curMP < 0:
            self.curMP = 0
        if self.curMP > self.stats["MP"]:
            self.curHP = self.stats["MP"]
            
    def setSP(self, value):
        self.curSP = value
        if self.curSP < 0:
            self.curSP = 0
        if self.curSP > 0:
            self.isAlive = True
        if self.curSP > self.stats["SP"]:
            self.curSP = self.stats["SP"]
        
    def addSP(self, value):
        if self.curSP > 0:
            self.curSP += value
        if self.curSP == 0:
            self.curSP = 0
        if self.curSP > self.stats["SP"]:
            self.curSP = self.stats["SP"]
            
    def addMove(self, moveID):
        if moveID not in self.moveList:
            self.moveList.append(moveID)
        else:
            print(f"[WARN]: Move {moveID} already exists for character {self.ID}")

    # Retorna os moves possíveis de acordo com o custo de SP e MP.
    def getMoveList(self):
        possibleMoves = []
        for move in self.moveList:
            if moveSheet[move]["SPCost"] <= self.curSP and moveSheet[move]["MPCost"] <= self.curMP:
                possibleMoves.append(move)
        return possibleMoves
    
    # Wrapper para todo gerenciamento de movimento
    def act(self, situation = []):
        moveList = self.getMoveList()
        return CombatAlgorithms.getMove(self.ID, self.isHero, situation, moveList, self.genome)

    # Retorna um dict simples de stats
    def dumpStats(self):
        exportStats = {
            "battleID" : self.ID,
            "typeID" : self.typeID,
            "isHero" : self.isHero,
            "Element" : self.stats["BaseElement"],
            "isAlive" : self.isAlive,
            "HP": self.curHP,
            "MaxHP": self.stats["HP"],
            "SP": self.curSP,
            "MaxSP": self.stats["SP"],
            "MP": self.curMP,
        }
        return exportStats

# Gerenciador de batalhas e o sistema RPG
class BattleManager:
    def __init__(self, verbose=False):
        self.turn = 0
        self.falseTurn = 0
        self.nextID = 0
        self.charList = []
        self.allyList = []
        self.foeList = []
        self.round = 0
        self.active_genome = None
        self.total_damage_dealt = 0
        self.verbose = verbose
        self.total_healed = 0
        
    def addHeroes(self, charID):    
        newHero = Character(charSheet[charID]["name"], charID, self.nextID, hero=True,
                            statSheet=charSheet[charID]["stats"],
                            moveList=charSheet[charID]["movelist"],
                            genome=self.active_genome)
        self.charList.append(newHero)
        self.allyList.append(self.nextID)
        self.nextID += 1
    
    def addFoes(self, charID):
        newFoe = Character(charSheet[charID]["name"], charID, self.nextID, hero=False,
                            statSheet=charSheet[charID]["stats"],
                            moveList=charSheet[charID]["movelist"])
        self.charList.append(newFoe)
        self.foeList.append(self.nextID)
        self.nextID += 1
    
    # Retorna como está o status da batalha atual
    def getBattleStatus(self):
        combatants = []
        for x in self.charList:
            combatants.append(x.dumpStats())
        return combatants
    
    # Pede e aplica um movimento para o personagem
    def requestMove(self):
        if self.verbose: self.print_status()
        
        # Quem deve ser o próximo a se movimentar
        arenaPlayers = len(self.charList)        
        nextMove = self.falseTurn % arenaPlayers
        while True:
            if self.charList[nextMove].isAlive:
                break
            self.falseTurn += 1
            nextMove = self.falseTurn % arenaPlayers
        
        # Pede o movimento para o personagem
        move, enemyList = self.charList[nextMove].act(self.getBattleStatus())
        
        if enemyList[0] == -1:
            print(f"[WARN] {self.charList[nextMove].name} não tem alvos.")
            self.turn += 1
            self.falseTurn += 1
            return
        
        # Para cada alvo válido
        for x in enemyList:
            self.applyMove(self.charList[nextMove].ID, x, move)
        
        # Recursos Consumidos
        self.charList[nextMove].addSP(-moveSheet[move]["SPCost"])
        self.charList[nextMove].addMP(-moveSheet[move]["MPCost"])
        
        # Regen passivo de SP para o heroi
        if self.charList[nextMove].isHero == True: self.charList[nextMove].addSP(0.2*self.charList[nextMove].stats["SP"])
        
    # Aplica o movimento
    def applyMove(self, userID, affectedID, move):
        isSelf = 1 if moveSheet[move]["Target"] == 1 else -1
        basePower = moveSheet[move]["BasePower"]
        moveName = moveSheet[move]["name"]
        
        attackerName = self.charList[userID].name
        targetName = self.charList[affectedID].name
        
        
        mult = CombatAlgorithms.getMultiplier(moveSheet[move]["Element"], self.charList[affectedID].stats["BaseElement"])
        statMult = 0
        damage = 0
        
        if isSelf == -1: # Se não deu self-target
            if moveSheet[move]["Type"] == 0: # Fisico
                statMult = self.charList[userID].stats["Str"]-self.charList[affectedID].stats["Def"]
            elif moveSheet[move]["Type"] == 1: # Ranged
                statMult = self.charList[userID].stats["Dex"]-self.charList[affectedID].stats["Def"]
            else: # Magico
                statMult = self.charList[userID].stats["Int"]-self.charList[affectedID].stats["Wis"]
        
            damage = basePower * mult * (1+(statMult/100))
            damage = max(1, int(damage)) # Garante min 1 de dano e inteiro
            
            affectedDMG = self.charList[affectedID].addHP(-damage)
            if self.charList[userID].isHero == True: self.total_damage_dealt -= affectedDMG

            if self.verbose:
                eff_text = ""
                if mult > 1.0: eff_text = "(Super Efetivo!)"
                elif mult < 1.0: eff_text = "(Pouco Efetivo...)"
                print(f"{attackerName} usou {moveName} em {targetName} >> {damage} dmg {eff_text}")

        else: # Se o move acerta o próprio usuario (Como Cura)
            heal_amount = self.charList[affectedID].stats["HP"] * (basePower/100) # Função de "dano" mais simples e sem influência elemental
            # Tracker de cura
            old_hp = self.charList[userID].curHP
            self.charList[userID].addHP(heal_amount)
            real_healed = self.charList[userID].curHP - old_hp
            if self.verbose:
                print(f"{attackerName} usou {moveName} e recuperou {real_healed} HP")
            if self.charList[userID].isHero:
                self.total_healed += real_healed
    
    # O GUI do jogo
    def print_status(self):
        if not self.verbose: return
        print("\n" + "="*40)
        print(f"--- TURNO {self.turn} ---")
        for char in self.charList:
            if not char.isAlive: continue
            hp_pct = char.curHP / char.stats["HP"]
            bars = int(hp_pct * 20)
            health_bar = "█" * bars + "░" * (20 - bars)
            icon = ":-)" if char.isHero else ">:C"
            print(f"{icon} {char.name[:10]:<10} [{health_bar}] {int(char.curHP)}/{char.stats['HP']}")
        print("="*40 + "\n")
        time.sleep(1.0) # Delay para cada turno
    
    # Limpa o cenário pra garantir novos rounds
    def cleanup(self):
        self.turn = 0
        self.falseTurn = 0
        self.nextID = 0
        self.total_damage_dealt = 0
        self.charList = []
        self.allyList = []
        self.foeList = []
        self.total_healed = 0
    
    # Wrapper de round simples, usado apenas para demonstração interna neste código
    def newRound(self):
        outcome, dmg, foeMaxHp, totalHealed = self.battleLoop()
        print(f"WIN: {dmg}/{foeMaxHp} dmg" if outcome == 1 else f"LOSE: {dmg}/{foeMaxHp} dmg")
        self.cleanup()
        self.round += 1
    
    # Se está todo mundo de um lado morto ou não
    def checkDeaths(self):
        heroAlive = False
        foeAlive = False
        for x in self.allyList:
            if self.charList[x].isAlive == True:
                heroAlive = True
        if not heroAlive: return -1 # LOSE
                
        for x in self.foeList:
            if self.charList[x].isAlive == True:
                foeAlive = True
        if not foeAlive: return 1 # WIN
        return 0 # Ainda está rolando
    
    # Loop principal de batalha, apenas para se um lado morrer inteiramente
    def battleLoop(self):
        self.addHeroes(0)
        numEnemies = random.randint(1, 3)
        
        bossRound = random.randint(1,10)
        if bossRound == 1:
            self.addFoes(7) # Ifrit
        else:
            for _ in range(numEnemies):
                self.addFoes(random.randint(1, 6))
            
        enemy_max_hp_total = 0
        for x in self.foeList:
            enemy_max_hp_total += self.charList[x].stats["HP"]
        
        while True:
            self.requestMove()
            battleStatus = self.checkDeaths()
            
            # Limite de segurança (para evitar loops infinitos de cura vs cura)
            if self.turn > 100: 
                return 0, self.total_damage_dealt, enemy_max_hp_total, self.total_healed # Empate por exaustão
            
            if battleStatus != 0: return battleStatus , self.total_damage_dealt, enemy_max_hp_total, self.total_healed
            self.turn += 1
            self.falseTurn += 1
          
if __name__ == "__main__":
    battle = BattleManager()
    battle.newRound()
    battle.newRound()
    battle.newRound()
    battle.newRound()