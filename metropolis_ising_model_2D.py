import numpy as np
import matplotlib.pyplot as plt
import numpy.random as random

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

class ChainGeneral2D:
    def __init__(self, N, M, J=[1.,1.], B = 0, mu = .5, k = 1, cold = True):
        self.Jx      = J[0]
        self.Jy      = J[1]
        self.B       = B
        self.mu      = mu
        self.k       = k
        self._N      = N # não alterar depois de iniciado
        self._M      = M # não alterar depois de iniciado
        self._lattice = self._start_config(cold) # somente new_config pode alterar

    def _start_config(self, cold):
        '''
        Retorna uma array N x M contendo os estados de spin inicial;
            * Se cold = True: teremos N x M spins 'Up' (+1)
            * Se cold = False: array  N x M aleatória de "Up's" & "Down's"
        '''
        N = self._N
        M = self._M

        if cold == True:
            lattice_ini = np.ones((N,M))
        else:
            lattice_ini = np.random.randint(-1, high=1, size=(N, M))
            lattice_ini += lattice_ini + 1
        return lattice_ini

    def thermalize(self, T, steps_therm = 10):
        '''
        "Termaliza" o sistema de N x M spins a uma temperatura T.
        Fazemos isso aplicando o algoritmo de Metropolis (steps_therm x N x M) vezes.
        '''
        N = self._N
        M = self._M
        for i in range(steps_therm * N * M):
            self.new_config(T)

    def calc_energy(self):
        '''
        Calcula a energia da cadeia de spins usando o Hamiltoniano do modelo de Ising
        '''
        μ = self.mu
        Jx = self.Jx
        Jy = self.Jy
        B = self.B
        S = self._lattice
        E = 0

        for i in range(self._N):
            for j in range(self._M):
                E += - Jx * S[i, j] * S[i, (j+1) % self._M ]\
                     - Jy * S[i, j] * S[(i+1) % self._N, j] \
                     - μ * S[i,j] * B
        return E

    def _calc_ΔE(self, indx, indy):
        '''
        Calcula a alteração na energia de um 'flip'.
            * Essa função não tem utilidade fora da classe.
            * Somente usar aqui dentro.
        '''
        S_t = self._lattice
        Jx  = self.Jx
        Jy  = self.Jy
        N   = self._N # indice maximo para indy
        M   = self._M # indice maximo para indx
        μ   = self.mu
        B   = self.B

        ΔE = -Jx * (S_t[indy, (indx+1) % M] + S_t[indy, indx-1]) \
             -Jy * (S_t[(indy+1) % N, indx] + S_t[indy-1, indx]) \
             -μ * B
        ΔE *= 2 * S_t[indy, indx]

        return ΔE

    def calc_magn(self):
        '''
        Calcula a magnetização normalizada pelo tamanho
        da cadeia de spins "N".
        '''
        return np.sum(self._lattice)/(self._N * self._M)

    def decide(self, indx, indy, T):

        '''
        Função para decidir se o estado teste ('flipado') é aceito, ou não,
        como novo estado de spins.
        '''
        S_t = self._lattice

        # Calcula Energia "atual" E_k:
        Delta_E = self._calc_ΔE(indx, indy)

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
        indx = random.randint(0, self._M) # índice x:
        indy = random.randint(0, self._N) # índice y:
        self._lattice[indy, indx] *= -1 # flip

        decision = self.decide(indx, indy, T) # Decisão:
        if decision == False:
            self._lattice[indy, indx] *= -1
        return None

    def plot_lattice(self):
        N       = self._N # size-y
        M       = self._M # size-x
        Spins   = self._lattice.reshape(N * M,)

        m_columns    = np.array(range(M)) + 1
        ones_columns = np.ones(M)

        n_rows     = np.array(range(1,N+1)).reshape(N,1)
        ones_rows  = np.ones(N).reshape(N, 1)

        Abscissa = (ones_rows * m_columns).reshape(N * M, )
        Ordenada = (n_rows * ones_columns).reshape(N * M, )
        Ordenada *= Spins

        Ups     = np.ones(N*M)
        Downs   = np.ones(N*M)

        for i in range(N*M):
            if Ordenada[i] > 0:
                Ups[i]   = Ordenada[i]
                Downs[i] = np.nan
            else:
                Ups[i]   = np.nan
                Downs[i] = -1. * Ordenada[i]

        fig = plt.figure(figsize = plt.figaspect(1.0))
        ax1 = fig.add_subplot(111)
        ax1.plot(Abscissa, Ups, "r^")
        ax1.plot(Abscissa, Downs, "bv")
        plt.show()

        return 0


def thermo_properties(lattice, T, rounds_metr = 100):
    '''
    Calcula médias num ensemble em equilíbrio com um
    banho térmico de temperatura 'T'.
    'M_rounds' é o tamanho da amostra no ensemble.

    Como output, esta função retorna
        - Energia média
        - Calor específico
        - Magnetização
    '''
    N = lattice._N
    M = lattice._M

    E_acumulada = 0 # Acumula energia
    E2_acumulada = 0 # Acumula energia ao quadrado
    M_acumulada = 0 # Acumula magnetização

    for i in range(rounds_metr * N * M):
        lattice.new_config(T)
        if i % (N*M) == (N*M-1):
            E_nova = lattice.calc_energy()
            E_acumulada += E_nova
            E2_acumulada += E_nova**2
            M_acumulada += lattice.calc_magn()


    E_media  = E_acumulada/rounds_metr
    E2_media = E2_acumulada/rounds_metr
    M_media  = M_acumulada/rounds_metr

    C_por_spin = (E2_media - E_media**2) / (lattice.k * T**2)

    return E_media, C_por_spin, M_media

def thermo_arrays(lattice, T_array, rounds_metr=10, steps_therm=10):
    """
    Esta função recebe, um objeto 'ChainGeneral2D', uma array de temperaturas
    e retorn três arrays com os valores de
        - Energia média
        - Calor específico
        - Magnetização
    para cada respectiva temperatura.
    """
    E_array = np.zeros(len(T_array))
    C_array = np.zeros(len(T_array))
    M_array = np.zeros(len(T_array))

    for i in range(len(T_array)):
        lattice.thermalize(T = T_array[i], steps_therm = steps_therm)
        E, C, M = thermo_properties(lattice, T_array[i], rounds_metr=rounds_metr)
        E_array[i], C_array[i], M_array[i] = E, C, M

    return E_array, C_array, M_array

def media_seeds(N_rows, M_columns, T_array, seeds=10, rounds_metr = 20, steps_therm = 20):
    '''
    Queremos diminuir a flutuação presente nos resultados
    obtidos para a energia, calor específico e magnetização.
    Para isso vamos executar a simulação algumas vezes com diferentes "seeds"
    e tomar a média ponto à ponto.

        * o argumento 'seeds' mostra quantas vezes faremos a simulação
    '''
    E_array = np.zeros(len(T_array)) # Acumula resultados p/ diferentes "seeds"
    C_array = np.zeros(len(T_array)) # Acumula resultados p/ diferentes "seeds"
    M_array = np.zeros(len(T_array)) # Acumula resultados p/ diferentes "seeds"

    for m in range(seeds):
        np.random.seed()
        lattice = ChainGeneral2D(N_rows, M_columns)
        # lattice._chain = lattice._start_config(cold = True)
        new_array_E, new_array_C, new_array_M = \
                thermo_arrays(lattice, T_array, rounds_metr = rounds_metr, steps_therm=steps_therm) # Calcula nova evolução de E e de C

        E_array += new_array_E # Acumula para calcular média
        C_array += new_array_C # Acumula para calcular média
        M_array += new_array_M
        # E_array_total[m, : ] = new_array_E # Salva novo resultado
        # C_array_total[m, : ] = new_array_C # Salva novo resultado

    E_array *= (1/seeds)
    C_array *= (1/seeds)
    M_array *= (1/seeds)

    return E_array, C_array, M_array


# Main:
N = int(input("N_rows [default = 10]: " ) or 10)
M = int(input("M_columns [default = 10]: ") or 10)
seeds = int(input("seeds [default = 5]: ") or 5)
rounds_metr = int(input("rounds_metr [default = 5]: ") or 5)
steps_therm = int(input("steps_therm [default = 5]: ") or 5)

T_array = np.linspace(0,10,100)[1:]

E_array, C_array, M_array = media_seeds(N, M, T_array, seeds=seeds,
                                                 rounds_metr=rounds_metr,
                                                 steps_therm=steps_therm)

fig = plt.figure(figsize=plt.figaspect(0.25))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
ax1.plot(T_array, E_array,linestyle='-', color='black' )
ax2.plot(T_array, C_array,linestyle='-', color='red')
ax3.plot(T_array, M_array,linestyle='-', color='maroon')

ax1.set_title(r"$\langle E \rangle$")
ax2.set_title(r"$C_V$")
ax3.set_title(r"$M$")

ax1.set_xlabel(r"$T[J/k_B]$")
ax2.set_xlabel(r"$T[J/k_B]$")
ax3.set_xlabel(r"$T[J/k_B]$")

ax1.set_ylabel(r"$E[J/k_B]$")
ax2.set_ylabel(r"$C_V[k_B]$")
ax3.set_ylabel(r"$M$")

plt.show()
