import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import sympy as sp

# valeurs des coefficients
c0=340
γ_p = 1.4
#mat q3
# φ = 0.99
# σ = 14000.0
# alpha_h = 1.02

#ITFH
# φ = 0.94
# σ = 9067
# alpha_h = 1

#ISOREL
φ = 0.7
σ = 142300
alpha_h = 1.15


def Simulation(φ,σ,alpha_h):
    ρ0 = 1.2
    L=1

    ξ0 =1 / (c0 ** 2)
    ξ1 = φ*γ_p/c0**2
    a= σ*φ**2*γ_p/(c0**2*ρ0*alpha_h)
    η0=1
    η1=φ/alpha_h
    w0=1640

    def e(k,g,ω,alpha):

        term1 = (1/np.sqrt(2))*np.sqrt(k**2-ξ1*ω**2/η1+np.sqrt((k**2-ξ1*ω**2/η1)**2+(a*ω/η1)**2))
        term2 = (1/np.sqrt(2))*np.sqrt(-k**2+ξ1*ω**2/η1+np.sqrt((k**2-ξ1*ω**2/η1)**2+(a*ω/η1)**2))
        λ1 = term1 - 1j * term2

        if k**2 < ξ0 * η0 * ω ** 2:
            λ0 = 1j*np.sqrt(-k ** 2 + ξ0 * η0 * ω ** 2)
            def f(x):
                return (λ0 * η0 - x) * np.exp(-λ0 * L) + (λ0 * η0 + x) * np.exp(λ0 * L)
            def χ(k, alpha):
                return g(k,ω) * ((λ0 * η0 - λ1 * η1) / f(λ1 * η1) - (λ0 * η0 - alpha) / f(alpha))
            def γ(k, alpha):
                return g(k,ω)* ((λ0 * η0 + λ1 * η1) / f(λ1 * η1) - (λ0 * η0 + alpha) / f(alpha))
            χ_value = χ(k, alpha)
            γ_value = γ(k, alpha)
            #expression de ek validée
            result = ((1 + k**2) * (L * (np.abs(χ_value)**2 + np.abs(γ_value)**2) + 1j /λ0 * np.imag(χ_value * np.conj(γ_value) * (1 - np.exp(-2 * λ0 * L))))
                            +L*np.abs(λ0)**2 * (np.abs(χ_value)**2 + np.abs(γ_value)**2) + 1j * λ0 * np.imag(χ_value * np.conj(γ_value) * (1 - np.exp(-2 * λ0 * L))))
            return np.real(result)

        if k**2 >= ξ0 * η0 * ω ** 2:
            λ0 = np.sqrt(k ** 2 - ξ0 * η0 * ω ** 2)
            def f(x):
                return (λ0 * η0 - x) * np.exp(-λ0 * L) + (λ0 * η0 + x) * np.exp(λ0 * L)
            def χ(k, alpha):
                return g(k,ω)* ((λ0 * η0 - λ1 * η1) / f(λ1 * η1) - (λ0 * η0 - alpha) / f(alpha))
            def γ(k, alpha):
                return g(k,ω)*((λ0 * η0 + λ1 * η1) / f(λ1 * η1) - (λ0 * η0 + alpha) / f(alpha))
            χ_value = χ(k, alpha)
            γ_value = γ(k, alpha)
            result = ((1 + k**2) * (1/(2 * λ0) * (1 - np.exp(-2 * λ0 * L)) * np.abs(χ_value)**2 + (np.exp(2 * λ0 * L) - 1)* np.abs(γ_value)**2 + 2 * L * np.real(χ_value * np.conj(γ_value)))
                    + 0.5*λ0*(1 - np.exp(-2 * λ0 * L)) * np.abs(χ_value)**2 + (np.exp(2 * λ0 * L) - 1)* np.abs(γ_value)**2)-2*λ0**2*L*np.real(χ_value * np.conj(γ_value))

            return np.real(result)

    def g_1(k,w):
        if k==0 : return 1
        return 0
    def g_2(k,w):
        def gr(y):
            return 80*np.exp(-(y/L) ** 2)*np.cos(k*y)
        def gi(y):
            return 80*np.exp(-(y/L) ** 2)*np.sin(k*y)
        r1,_ = quad(lambda y: gr(y), -L, L)
        r2, _ = quad(lambda y: gi(y), -L, L)
        return r1+1j*r2
    def g_3(k,w):
        if k==np.pi/L: return -1/2j
        if k==-np.pi/L : return 1/2j
        return 0
    #Simulation
    R_alpha = []
    I_alpha = []
    #Range_omega = range(600,650) # définition de l'intervalle de variation de omega
    Range_omega = range(1256,6000) # définition de l'intervalle de variation de omega
    #Le but est de minimiser la somme(alpha) Z pour alpha complexe
    for w in Range_omega:
        def somme(alpha, w): # calcul de la somme à minimiser
            s = 0
            for n in range(-30, 30):# la somme allant de -50 à 50 est suffisante aprés quelques tests
                k = n * np.pi / L
                s += e(k, g_1, w, alpha)
            return np.real(s)
        def f(x, y): #on écrit la somme comme fonction de x et y
            return somme(x + y * 1j,w)
        #on cherche le minimum de f avec la méthode BFGS, qui renvoie donc le alpha qui minimise cette somme
        result = minimize(lambda x: f(x[0], x[1]), x0=[1, 1],method='L-BFGS-B')
        R_alpha.append(result.x[0]) #partie réelle de alpha
        I_alpha.append(result.x[1]) #partie imaginaire de alpha


    #tracé des courbes
    # frequences=np.array(Range_omega)*(1/(2*np.pi))
    # plt.scatter(frequences,R_alpha,s=1)
    # plt.xlabel('Range_omega')
    # plt.ylabel('Re(alpha)')
    # plt.show()
    # plt.scatter(frequences,I_alpha, s=1)
    # plt.xlabel('Range_omega')
    # plt.ylabel('Im(alpha)')
    # plt.show()
    return R_alpha,I_alpha



def plot_Im_over_Re_for_all_materials():
    Range_omega = range(1256,6000)
    frequences=np.array(Range_omega)*(1/(2*np.pi))
    #mat q3
    φ = 0.99
    σ = 14000.0
    alpha_h = 1.02

    R_alpha_q3,I_alpha_q3=Simulation(φ,σ,alpha_h)
    q3=np.array(R_alpha_q3)/np.array(I_alpha_q3)
    plt.plot(frequences,q3,label='q3')
    plt.xlabel('Range_f')
    plt.ylabel('Re(alpha)/Im(alpha)')
    plt.legend()
    plt.savefig('C:\\Users\\Farouk\\Desktop\\ST5_ei\\Re(alpha) over Im(alpha)_q3.png')
    #plt.show()

    #ITFH
    φ = 0.94
    σ = 9067
    alpha_h = 1

    R_alpha_IFTH,I_alpha_IFTH=Simulation(φ,σ,alpha_h)
    IFTH=np.array(R_alpha_IFTH)/np.array(I_alpha_IFTH)
    plt.plot(frequences,IFTH,label='IFTH')
    plt.xlabel('Range_f')
    plt.ylabel('Re(alpha)/Im(alpha)')
    plt.legend()
    #plt.show()
    plt.savefig('C:\\Users\\Farouk\\Desktop\\ST5_ei\\Re(alpha) over Im(alpha)_IFTH.png')

    #ISOREL
    φ = 0.7
    σ = 142300
    alpha_h = 1.15

    R_alpha_ISOREL,I_alpha_ISOREL=Simulation(φ,σ,alpha_h)
    ISOREL=np.array(R_alpha_ISOREL)/np.array(I_alpha_ISOREL)
    plt.plot(frequences,ISOREL,label='ISOREL')
    plt.xlabel('Range_f')
    plt.ylabel('Re(alpha)/Im(alpha)')
    plt.legend()
    plt.savefig('C:\\Users\\Farouk\\Desktop\\ST5_ei\\Re(alpha) over Im(alpha)_ISOREL.png')
    #plt.show()
    return
plot_Im_over_Re_for_all_materials()