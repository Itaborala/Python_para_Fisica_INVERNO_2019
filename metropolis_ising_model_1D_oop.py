import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

'''
Talvez para analisar as propriedades termodinâmicas seja melhor
usar uma class 'Chain' para facilitar as coisas:
'''

class ChainGeneral:
    def __init__(self, N, J=.1, B = 0, mu = .5, k = 1, cold = True):
        self._N     = N # não alterar depois de iniciado
        self.J      = J
        self.B      = B
        self.mu     = mu
        self.k      = k
        self._chain = self._start_config(cold) # somente new_config pode alterar

    def _start_config(self, cold):
        '''
        Retorna uma array contendo os estados de spin inicial;
            * Se cold = True: teremos N spins 'Up' (+1)
            * Se cold = False: array aleatória de "Up's" & "Down's"
        '''
        N = self._N

        if cold == True:
            chain_ini = np.ones(N)
        else:
            chain_ini = np.random.randint(-1, high=1, size=N)
            chain_ini += chain_ini + 1
        return chain_ini

    def thermalize(self, T, steps = 10):
        '''
        "Termaliza" o sistema de 'N' spins a uma temperatura T.
        Fazemos isso aplicando o algoritmo de Metropolis (steps x N) vezes.
        '''
        N = self._N
        for i in range(steps * N):
            self.new_config(T)

    def calc_energy(self):
        '''
        Calcula a energia da cadeia de spins usando o Hamiltoniano do modelo de Ising
        '''
        μ = self.mu
        J = self.J
        B = self.B
        S = self._chain
        E = 0
        for i in range(len(S)-1):
            E += - J * S[i]*S[i+1] - μ * S[i] * B
        E += μ * S[-1] * B # contabiliza último sítio
        return E

    def _calc_ΔE(self, ind):
        '''
        Calcula a alteração na energia de um 'flip'.
            * Essa função não tem utilidade fora da classe.
            * Somente usar aqui dentro.
        '''
        S_t = self._chain
        J   = self.J
        μ   = self.mu
        B   = self.B

        i_0 = ind - 1
        i_2 = ind + 1

        if ind == 0:
            ΔE = 2 * S_t[ind] * (-J * S_t[i_2] - μ*B)
        elif ind == len(S_t) - 1:
            ΔE = 2 * S_t[ind] * (-J * S_t[i_0] - μ*B)
        else:
            ΔE = 2 * S_t[ind] * (-J * (S_t[i_2] + S_t[i_0]) - μ*B)
        return ΔE

    def calc_magn(self):
        '''
        Calcula a magnetização normalizada pelo tamanho
        da cadeia de spins "N".
        '''
        return np.sum(self._chain)/self._N

    def decide(self, ind, T):

        '''
        Função para decidir se o estado teste ('flipado') é aceito, ou não,
        como novo estado de spins.
        '''
        S_t = self._chain

        # Calcula Energia "atual" E_k:
        Delta_E = self._calc_ΔE(ind)

        if Delta_E <= 0:
            return True
        else:
            R = np.exp(-Delta_E/(self.k * T))
            rᵢ = random.uniform()
            if rᵢ <= R:
                return True
            else:
                return False

    def new_config(self, T):
        '''
        Executa uma 'rodada' do algoritmo de Metropolis.
        '''
        ind = random.randint(0, self._N) # Escolhe um sítio:
        self._chain[ind] *= -1 # flip

        decision = self.decide(ind, T) # Decisão:
        if decision == False:
            self._chain[ind] *= -1

        return None


def average_energy_specific_heat(chain, T, rounds=100):
    '''
    Calcula a média num ensemble em equilíbrio com um
    banho térmico de temperatura 'T'.
    'rounds' é o tamanho da amostra no ensemble.
    '''
    N = chain._N
    E_acumulada = 0
    E2_acumulada = 0

    for i in range(rounds * N):
        chain.new_config(T)
        if i % N == (N-1):
            E_nova = chain.calc_energy()
            E_acumulada += E_nova
            E2_acumulada += E_nova**2


    E_media = E_acumulada/rounds
    E2_media = E2_acumulada/rounds

    C_por_spin = (E2_media - E_media**2) / (chain.k * T**2)

    return E_media, C_por_spin

def energy_array(ising_chain, T_array, rounds_metr = 10, steps_therm = 10):
    '''
    Gera uma array de valores de Energia dado uma cadeia de Ising e
    uma array contendo os valores de temperatura.
    '''
    E_array = np.zeros(len(T_array))
    C_array = np.zeros(len(T_array))

    for i in range(len(T_array)):
        ising_chain.thermalize( T = T_array[i], steps = steps_therm)
        E_array[i], C_array[i] =  average_energy_specific_heat(ising_chain,
                                    T = T_array[i], rounds = rounds_metr)

    return E_array, C_array

def media_seeds(ising_chain, T_array, seeds=10):
    '''
    Queremos diminuir a flutuação presente nos resultados
    obtidos para a energia, calor específico e magnetização.
    Para isso vamos executar a simulação algumas vezes com diferentes "seeds"
    e tomar a média ponto à ponto.

        * o argumento 'seeds' mostra quantas vezes faremos a simulação
    '''
    E_array = np.zeros(len(T_array)) # Acumula resultados p/ diferentes "seeds"
    C_array = np.zeros(len(T_array)) # Acumula resultados p/ diferentes "seeds"

    for m in range(seeds):
        np.random.seed()
        ising_chain._chain = ising_chain._start_config(cold = True)
        new_array_E, new_array_C = \
                energy_array(ising_chain, T_array, rounds_metr = 20) # Calcula nova evolução de E e de C

        E_array += new_array_E # Acumula para calcular média
        C_array += new_array_C # Acumula para calcular média

        # E_array_total[m, : ] = new_array_E # Salva novo resultado
        # C_array_total[m, : ] = new_array_C # Salva novo resultado

    E_array *= (1/seeds)
    C_array *= (1/seeds)

    return E_array, C_array



T_array = np.linspace(0,4,401)[1:]

N_spins = 200
ising_chain = ChainGeneral(N=N_spins, J=1.)

M_seeds = 10
E_array, C_array = media_seeds(ising_chain, T_array, seeds=M_seeds)

## Cada linha da array "E_array_total" corresponde a um "seed" diferente,
## enquanto que cada coluna corresponde a uma temperatura diferente.
# E_array_total = np.zeros(( M_seeds, len(T_array) ) )
# C_array_total = np.zeros(( M_seeds, len(T_array) ) )
# np.save('Energy_200_spins_100_seeds.npy', E_array_total)
# np.save('C_200_spins_100_seeds.npy', C_array_total)

J = ising_chain.J
k_B = ising_chain.k
E_analitico = -J * N_spins * np.tanh(J/T_array)
C_analitico = N_spins * (J/(k_B*T_array))**2 / (np.cosh(J/(k_B*T_array)))**2

fig = plt.figure(figsize=plt.figaspect(0.25))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.plot(T_array, E_array)
ax2.plot(T_array, C_array)
ax3.plot(T_array, E_analitico)
ax4.plot(T_array, C_analitico)
plt.show()
