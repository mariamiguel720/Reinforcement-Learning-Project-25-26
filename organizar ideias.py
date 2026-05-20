"""
Config A:
    - Policy Iteration (Cap. 3)
    - Monte Carlo (Cap. 3)
    - SARSA (Cap. 4)
    - Q-Learning (Cap. 4)

Config B:
    como chave do dicionário.  (fraco, pode ser substituido por um algoritmo melhor que dermos)
    - DQN (Deep Q-Network) — é a extensão directa de Q-Learning para espaços contínuos, substituindo a Q-table por uma rede neural. 
    - PPO é uma boa opção mas mais avançada — pertence à família actor-critic (policy-based + value-based)

DQN é value-based, off-policy, aprende Q(s,a)
PPO é policy-based, on-policy, aprende π(a|s) directamente



Esperem pelas aulas de function approximation para confirmar o que vai ser dado, mas planeiem com Monte Carlo 
agora + DQN quando as aulas chegarem lá. Se a professora ainda der actor-critic, podem substituir MC por PPO 
como segundo algoritmo — ficaria ainda mais forte para o relatório.