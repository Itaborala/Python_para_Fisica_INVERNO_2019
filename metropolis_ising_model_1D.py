import numpy as np
import matplotlib.pyplot as plt
from numpy import random


def start_config(N, cold = True):
    '''
    Retorna uma array contendo os estados de spin inicial;
        * Se cold = True: teremos N spins 'Up' (+1)
        * Se cold = False: array aleatória de "Up's" (+1) & "Down's" (-1)
    '''
    if cold == True:
        chain_ini = np.ones(N)
    else:
        chain_ini =  np.random.randint(-1, high=1, size=N)
        chain_ini += chain_ini + 1
    return chain_ini

def calc_energy(S, J = .1, B = 0, μ = .5):
    '''
    Calcula a energia da cadeia de spins usando o Hamiltoniano do modelo de Ising
    '''
    E = 0
    for i in range(len(S)-1):
        E += - J * S[i]*S[i+1] - μ * S[i] * B
    E += μ * S[-1] * B # contabiliza último sítio
    return E

def calc_ΔE(S_t, ind, J = .1, B = 0, μ = .5):
    '''
    Calcula a alteração na energia de um 'flip'
    '''
    i_0 = ind - 1
    i_2 = ind + 1

    if ind == 0:
        ΔE = 2 * S_t[ind] * (-J * S_t[i_2] - μ*B)
    elif ind == len(S_t) - 1:
        ΔE = 2 * S_t[ind] * (-J * S_t[i_0] - μ*B)
    else:
        ΔE = 2 * S_t[ind] * (-J * (S_t[i_2] + S_t[i_0]) - μ*B)
    return ΔE

def decide(S_t, ind, T):
    '''
    Função para decidir se o estado teste ('flipado') é aceito, ou não,
    como novo estado de spins.
    '''

    Delta_E = calc_ΔE(S_t, ind)
    if Delta_E <= 0:
        return True
    else:
        R = np.exp(- Delta_E / T)
        rᵢ = random.uniform()
        if rᵢ <= R:
            return True
        else:
            return False

def new_config(S_k, k = 1, T = 0.0001):
    '''
    Sorteia um sítio aleatório e testa usando a função 'decide'.
    Se o sorteio não for aceito, retorne a cadeia inicial,
    caso contrário retorne a nova cadeia 'flipada'.
    '''
    # Escolhe um sítio:
    ind = random.randint(0, len(S_k))

    # Config. teste:
    S_k[ind] *= -1 # flip

    # Decisão:
    decision = decide(S_k, ind, T)
    if decision == False:
        S_k[ind] *= -1
    return S_k

def energy_media_T(alpha, N, M_rounds, T):

    E_acumulada = 0
    for i in range(M_rounds * N):
        alpha = new_config(alpha,T = T)
        if i % N == (N-1):
            # somente depois de um número considerável 
            # de tentativas (comparável ao número de spins),
            # considere o valor da energia na média.
            E_acumulada += calc_energy(alpha)

    return E_acumulada / M_rounds


N = 200 # tamanho da cadeia
M = 20  # rounds de Monte-Carlo
J = .1  # valor da "integral de troca"
alpha = start_config(N, cold = True)

T_array = np.linspace(0,1,100)[1:]
E_array = np.array([energy_media_T(alpha, N, M, T) for T in T_array])
E_analitico = -J * N * np.tanh(J/T_array)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

ax1.plot(T_array, E_array)
ax1.plot(T_array, E_analitico)

################################################################################
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)

# ax2.plot(T_array, E_array)
# ax2.plot(T_array, E_analitico)

plt.show()

# print('\nThermalized state: \n', alpha)
