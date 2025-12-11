# Jogador RPG Evolutivo

> Implementação de um Algoritmo Evolutivo aplicado a um sistema de combate por turnos (JRPG).

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)

## Sobre o Projeto

Este projeto foi desenvolvido para a disciplina de SSC0713: Sistemas Evolutivos Aplicados a Robotica.

[Link do Vídeo Explicativo](Ainda a ser postado versão 2)

O objetivo foi criar um sistema evolutivo capaz de aprender, do zero, a jogar um RPG de turno estilo clássico.

O agente utiliza uma Rede Neural como cérebro e é treinado através de um Algoritmo Genético. Ao longo do treinamento, a IA começa a gerenciar melhor os recursos (SP/MP),  explorar fraquezas elementais e decidir o momento exato de curar ou atacar para maximizar a taxa de vitórias.

**Resultado Final:** O agente alcançou uma taxa de vitória de aproximadamente **~95%** contra inimigos procedurais e Bosses, partindo de um comportamento aleatório.

## Funcionamento

A IA toma decisões baseada em um vetor de inputs normalizados que passam por uma rede neural.

### 1. Inputs (Sensores)
A parte mais importante está nos dados de entrada escolhidos para serem passados, especificamente o **"Pain Level"**:
* **Pain Level (Nível de Dor):** Input de HP (`1.0 - HP_Atual`). Permite que a IA "sinta" o risco de morte com mais intensidade do que apenas observar a barra de vida absoluta.
* **Vantagem Elemental:** Mapeamento de fraquezas/resistências (-1.0 a 1.0).
* **Gestão de Recursos:** Custo do golpe vs. SP disponível.

### 2. Função de Fitness (Reward Shaping)
Para evitar comportamentos suicidas ou passivos, a função de recompensa avalia:
* **Dano Causado:** Incentivo forte, mas não crucial.
* **Vitória/Derrota:** Recompensa massiva ou punição forte.
* **Cura Efetiva:** Pontos por recuperar HP perdido em situações críticas.
* **Punição por "Burrice":** Penalidade severa se morrer sendo que tinha alguma forma de se curar.

### 3. Mecânica Evolutiva
* **População, Geração e Rodadas:** 50-100 indivíduos.
* **Seleção:** Torneio + Elitismo (Top 3 mantidos intocados).
* **Mutação:** Taxa adaptativa (decaimento linear) para refinar o comportamento nas gerações finais e evitar perda de traços desejáveis.

---

## Instalação e Execução

### Pré-requisitos
Certifique-se de ter o Python instalado. Instale as dependências:

```bash
pip install numpy matplotlib
```

### Treinando o Modelo
Criando um novo genoma campeão:

```bash
python train.py
```

### Assistindo o Melhor Genoma
Ver o resultado do modelo em tempo real com output do combate:

```bash
python replay.py
```