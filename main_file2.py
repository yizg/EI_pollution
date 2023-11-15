# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
import descente_grad
# import solutions


def final_projected(chi, beta):
    # find the beta th largest value of chi
    chi_copy = chi.copy()
    chi_copy = chi_copy.reshape(-1)
    chi_copy.sort()
    chi_copy = chi_copy[::-1]
    threshold = chi_copy[beta-1]
    # set to zero all the values of chi that are smaller than the threshold
    chi[chi < threshold] = 0.
    # set to one all the values of chi that are greater than the threshold
    chi[chi >= threshold] = 1.
    print(numpy.sum(numpy.sum(chi)))
    return chi

def your_compute_objective_function(domain_omega, u, spacestep):
    """
    This function compute the objective function:
    J(u,domain_omega)= \int_{domain_omega}||u||^2 

    Parameter:
        domain_omega: Matrix (NxP), it defines the domain and the shape of the
        Robin frontier;
        u: Matrix (NxP), it is the solution of the Helmholtz problem, we are
        computing its energy;
        spacestep: float, it corresponds to the step used to solve the Helmholtz
        equation.
    """

    # mask = domain_omega == _env.NODE_INTERIOR
    # energy = numpy.sum(numpy.sum(numpy.abs(u[mask])**2))*spacestep**2

    (M, N) = numpy.shape(domain_omega)
    spacestep_x = 1.0 / N
    spacestep_y = 1.0 / M

    energy = 0.0

    for i in range(0, M-1):
        for j in range(0, N-1):
            if domain_omega[i, j] == _env.NODE_INTERIOR or domain_omega[i+1, j] == _env.NODE_INTERIOR or domain_omega[i, j+1] == _env.NODE_INTERIOR or domain_omega[i+1, j+1] == _env.NODE_INTERIOR:
                energy += (numpy.abs(u[i, j])**2 + numpy.abs(u[i+1, j])**2 +
                           numpy.abs(u[i, j+1])**2 + numpy.abs(u[i+1, j+1])**2)*spacestep_x*spacestep_y/4

    return energy


def compute_projected(chi, domain, V_obj):
    """This function performs the projection of $\chi^n - mu*grad

    To perform the optimization, we use a projected gradient algorithm. This
    function caracterizes the projection of chi onto the admissible space
    (the space of $L^{infty}$ function which volume is equal to $V_{obj}$ and whose
    values are located between 0 and 1).

    :param chi: density matrix
    :param domain: domain of definition of the equations
    :param V_obj: characterizes the volume constraint
    :type chi: numpy.array((M,N), dtype=float64)
    :type domain: numpy.array((M,N), dtype=complex128)
    :type float: float
    :return:
    :rtype:
    """

    (M, N) = numpy.shape(domain)
    S = 0
    for i in range(M):
        for j in range(N):
            if domain[i, j] == _env.NODE_ROBIN:
                S = S + 1

    B = chi.copy()
    l = 0
    chi = preprocessing.set2zero(chi, domain)

    V = numpy.sum(numpy.sum(chi)) / S
    debut = -numpy.max(chi)
    fin = numpy.max(chi)
    ecart = fin - debut
    # We use dichotomy to find a constant such that chi^{n+1}=max(0,min(chi^{n}+l,1)) is an element of the admissible space
    while ecart > 10 ** -4:
        # calcul du milieu
        l = (debut + fin) / 2
        for i in range(M):
            for j in range(N):
                chi[i, j] = numpy.maximum(0, numpy.minimum(B[i, j] + l, 1))
        chi = preprocessing.set2zero(chi, domain)
        V = sum(sum(chi)) / S
        if V > V_obj:
            fin = l
        else:
            debut = l
        ecart = fin - debut
        # print('le volume est', V, 'le volume objectif est', V_obj)

    return chi


def BelongsInteriorDomain(node):
	if (node < 0):
		return 1
	if node == 3:
		# print("Robin")
		return 2
	else:
		return 0


def compute_gradient_descent(chi, grad, domain, mu):
	"""This function makes the gradient descent.
	This function has to be used before the 'Projected' function that will project
	the new element onto the admissible space.
	:param chi: density of absorption define everywhere in the domain
	:param grad: parametric gradient associated to the problem
	:param domain: domain of definition of the equations
	:param mu: step of the descent
	:type chi: numpy.array((M,N), dtype=float64
	:type grad: numpy.array((M,N), dtype=float64)
	:type domain: numpy.array((M,N), dtype=int64)
	:type mu: float
	:return chi:
	:rtype chi: numpy.array((M,N), dtype=float64

	.. warnings also: It is important that the conditions be expressed with an "if",
			not with an "elif", as some points are neighbours to multiple points
			of the Robin frontier.
	"""

	(M, N) = numpy.shape(domain)
	# for i in range(0, M):
	# 	for j in range(0, N):
	# 		if domain_omega[i, j] != _env.NODE_ROBIN:
	# 			chi[i, j] = chi[i, j] - mu * grad[i, j]
	# # for i in range(0, M):
	# 	for j in range(0, N):
	# 		if preprocessing.is_on_boundary(domain[i , j]) == 'BOUNDARY':
	# 			chi[i,j] = chi[i,j] - mu*grad[i,j]
	# print(domain,'jesuisla')
	#chi[50,:] = chi[50,:] - mu*grad[50,:]
	for i in range(1, M - 1):
		for j in range(1, N - 1):
			#print(i,j)
			#chi[i,j] = chi[i,j] - mu * grad[i,j]
			a = BelongsInteriorDomain(domain[i + 1, j])
			b = BelongsInteriorDomain(domain[i - 1, j])
			c = BelongsInteriorDomain(domain[i, j + 1])
			d = BelongsInteriorDomain(domain[i, j - 1])
			if a == 2:
				# print(i+1,j, "-----", "i+1,j")
				chi[i + 1, j] = chi[i + 1, j] - mu * grad[i, j]
			if b == 2:
				# print(i - 1, j, "-----", "i - 1, j")
				chi[i - 1, j] = chi[i - 1, j] - mu * grad[i, j]
			if c == 2:
				# print(i, j + 1, "-----", "i , j + 1")
				chi[i, j + 1] = chi[i, j + 1] - mu * grad[i, j]
			if d == 2:
				# print(i, j - 1, "-----", "i , j - 1")
				chi[i, j - 1] = chi[i, j - 1] - mu * grad[i, j]

	return chi


def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                Alpha, mu, chi, V_obj,beta):
    """This function return the optimized density.

    Parameter:
        cf solvehelmholtz's remarks
        Alpha: complex, it corresponds to the absorbtion coefficient;
        mu: float, it is the initial step of the gradient's descent;
        V_obj: float, it characterizes the volume constraint on the density chi.
    """
    g_dir = numpy.zeros(f_dir.shape)
    g_neu = numpy.zeros(f_neu.shape)
    g_rob = numpy.zeros(f_rob.shape)
    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 10
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    while k < numb_iter and mu > 10**(-5):
        #print('---- iteration number = ', k)
        # print('1. computing solution of Helmholtz problem, i.e., u')
        u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        # print('2. computing solution of adjoint problem, i.e., p')

        q = processing.solve_helmholtz(domain_omega, spacestep, omega, -2*numpy.conjugate(u), g_dir, g_neu, g_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        # print('3. computing objective function, i.e., energy')
        energy[k] = your_compute_objective_function(domain_omega, u, spacestep)
        ene = energy[k]
        # print('4. computing parametric gradient')

        grad =  - numpy.real(Alpha*u*q)

        while ene >= energy[k] and mu > 10 ** -5:
            #print('dans la deuxième boucle de ', k,'on est à ',c)
            # print('    a. computing gradient descent')
            chi = descente_grad.compute_gradient_descent(chi, grad, domain_omega, mu)
            #chi=compute_gradient_descent(chi, grad, domain_omega, mu)
            
            #print('chiiiiiii: ', chi)
            # print('    b. computing projected gradient')
            chi = compute_projected(chi, domain_omega, V_obj)

            chi=preprocessing.set2zero(chi,domain_omega)
            # chi=chi-mu*grad
            # chi=preprocessing.set2zero(chi,domain_omega)
            #print('norm chi', numpy.linalg.norm(chi,1))
            # print('    c. computing solution of Helmholtz problem, i.e., u')
            alpha_rob = Alpha*chi
            u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            # q = processing.solve_helmholtz(domain_omega, spacestep, omega, -2*numpy.conjugate(u), g_dir, g_neu, g_rob,
            #                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            # grad = - numpy.real(Alpha*u*q)
            # print('    d. computing objective function, i.e., energy (E)')
            ene = your_compute_objective_function(domain_omega, u, spacestep)
            bool_a = ene < energy[k]
            if bool_a:
                # The step is increased if the energy decreased
                mu = mu * 1.1
            else:
                # The step is decreased is the energy increased
                mu = mu / 2
        print("mu=" + str(mu), "ene=" + str(ene), "grad=" + str(numpy.linalg.norm(grad, 2)))
        k += 1

    # print('end. computing solution of Helmholtz problem, i.e., u')
    chi = final_projected(chi,beta)
    alpha_rob = Alpha*chi
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    energy = energy[:k]
    energy[-1]=your_compute_objective_function(domain_omega,u,spacestep)
    return chi, energy, u, grad

def last_non_null_value(L):
    for k in range(len(L)-1,-1,-1):
         if L[k]!=0: 
              return L[k]
        
def plot_energy_optimized_of_frequencies(Alpha,level):
    #f entre 100 ET 10 000 
    #k entre (1,85) et 185
    wave_numbers=numpy.array([k for k in range(184,186)])
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    spacestep = 1.0 / N  # mesh size
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(
        M, N)
    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(
        M, N, level)
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # -- initialize
    #Start computing energies for each wavenumber k
    Energies=[]
    for k in wave_numbers:
        alpha_rob[:, :] = - k * 1j
        # -- define material density matrix
        chi = preprocessing._set_chi(M, N, x, y)
        chi = preprocessing.set2zero(chi, domain_omega)
        nb_pixels = int(numpy.sum(numpy.sum(chi)))
        # -- define absorbing material
        # Alpha = compute_alpha.compute_alpha(...)
        alpha_rob = Alpha * chi

        # -- set parameters for optimization
        S = 0  # surface of the fractal
        for i in range(0, M):
            for j in range(0, N):
                if domain_omega[i, j] == _env.NODE_ROBIN:
                    S += 1
        V_0 = 1  # initial volume of the domain
        V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
        mu = 5  # initial gradient step
        energy=last_non_null_value(your_optimization_procedure(domain_omega, spacestep, k, f, f_dir, f_neu, f_rob,
                            beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                            Alpha, mu, chi, V_obj,nb_pixels)[1])
        Energies.append(energy)
    
    matplotlib.pyplot.plot(wave_numbers,Energies)  
    matplotlib.pyplot.show()
    return 
#Day two 
#First, for a fully absorbant boundary ie chi=1 on Robin
def plot_energy_of_frequencies(Alpha,level):
    wave_numbers=numpy.array([k for k in range(1,186)])
    frequencies=wave_numbers*(340/(2*numpy.pi))
    N = 50# number of points along x-axis
    M = 2 * N  # number of points along y-axis
    spacestep = 1.0 / N  # mesh size
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(
        M, N)
    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)
    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(
        M, N, level)
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # -- initialize
    #Start computing energies for each wavenumber k
    Energies=[]
    for k in wave_numbers:
        alpha_rob[:, :] = - k * 1j
        # -- define material density matrix
        chi = preprocessing._set_chi(M, N, x, y)
        chi = preprocessing.set2zero(chi, domain_omega)
        ones=numpy.ones(chi.shape)
        # -- define absorbing material
        # Alpha = compute_alpha.compute_alpha(...)
        alpha_rob = Alpha * chi
        S = 0  # surface of the fractal
        for i in range(0, M):
            for j in range(0, N):
                if domain_omega[i, j] == _env.NODE_ROBIN:
                    S += 1
        V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
        mu = 5  # initial gradient step
        u=processing.solve_helmholtz(domain_omega,spacestep,k,f,f_dir,f_neu,f_rob,beta_pde,alpha_pde,alpha_dir,beta_neu,beta_rob,alpha_rob)
        energy=your_compute_objective_function(domain_omega,u,spacestep)
        Energies.append(energy)
    
    # matplotlib.pyplot.plot(frequencies,Energies)
    # matplotlib.pyplot.show()
    return Energies,domain_omega, spacestep, f, f_dir, f_neu, f_rob,beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,mu, chi, V_obj


def plot_first_vs_optimized(Alpha,level):
    wavenumbers=numpy.array([k for k in range(1,186)])
    Energies=plot_energy_of_frequencies(Alpha,level)[0]
    E1=sorted(list(enumerate(Energies)),key= lambda x:x[1])[-7:]
    domain_omega, spacestep, f, f_dir, f_neu, f_rob,beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,mu, chi, V_obj=plot_energy_of_frequencies(Alpha,level)[1:]

    indices_k=[]
    non_op_energies=[]
    #old_energies=[x[1] for x in E1]
    new_energies=[]
    alpha_rob=Alpha*chi
    nb_pixels = int(numpy.sum(numpy.sum(chi)))
    for x in E1:
         k,non_op_ene=wavenumbers[x[0]],x[1]

         indices_k.append(k*float(340/(2*numpy.pi)))
         non_op_energies.append(non_op_ene)
         new_energies.append(last_non_null_value(your_optimization_procedure(domain_omega, spacestep,k, f, f_dir, f_neu, f_rob,beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,Alpha,mu, chi, V_obj,nb_pixels)[1]))
    matplotlib.pyplot.scatter(indices_k,non_op_energies,color='red',label='non optimized')
    matplotlib.pyplot.scatter(indices_k,new_energies,color='blue',label='optimized')
    matplotlib.pyplot.xlabel('Fréquences d energie maximale')
    matplotlib.pyplot.ylabel('Energies')

    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


Alpha=10-0.1*1j        
#plot_first_vs_optimized(Alpha,0)
plot_first_vs_optimized(Alpha,1)
    
    
    
    
# if __name__ == '__main__':

#     # ----------------------------------------------------------------------
#     # -- Fell free to modify the function call in this cell.
#     # ----------------------------------------------------------------------
#     # -- set parameters of the geometry
#     N = 50  # number of points along x-axis
#     M = 2 * N  # number of points along y-axis
#     level = 2  # level of the fractal
#     spacestep = 1.0 / N  # mesh size

#     # -- set parameters of the partial differential equation
#     kx = -1.0
#     ky = -1.0
#     wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
#     wavenumber = 10

#     # ----------------------------------------------------------------------
#     # -- Do not modify this cell, these are the values that you will be assessed against.
#     # ----------------------------------------------------------------------
#     # --- set coefficients of the partial differential equation
#     beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(
#         M, N)

#     # -- set right hand sides of the partial differential equation
#     f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

#     # -- set geometry of domain
#     domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(
#         M, N, level)

#     # ----------------------------------------------------------------------
#     # -- Fell free to modify the function call in this cell.
#     # ----------------------------------------------------------------------
#     # -- define boundary conditions
#     # planar wave defined on top
#     f_dir[:, :] = 0.0
#     f_dir[0, 0:N] = 1.0
#     # spherical wave defined on top
#     # f_dir[:, :] = 0.0
#     # f_dir[0, int(N/2)] = 10.0

#     # -- initialize
#     alpha_rob[:, :] = - wavenumber * 1j

#     # -- define material density matrix
#     chi = preprocessing._set_chi(M, N, x, y)
#     chi = preprocessing.set2zero(chi, domain_omega)
#     nb_pixels = int(numpy.sum(numy.sum(chi)))
#     print(chi.shape)
#     # -- define absorbing material
#     Alpha = 10.0 - 10.0 * 1j
#     # -- this is the function you have written during your project
#     # import compute_alpha
#     # Alpha = compute_alpha.compute_alpha(...)
#     alpha_rob = Alpha * chi

#     # -- set parameters for optimization
#     S = 0  # surface of the fractal
#     for i in range(0, M):
#         for j in range(0, N):
#             if domain_omega[i, j] == _env.NODE_ROBIN:
#                 S += 1
#     V_0 = 1  # initial volume of the domain
#     V_obj = numpy.sum(numpy.sum(chi)) / S  # constraint on the density
#     print('V_obj=' + str(V_obj))
#     mu = 5  # initial gradient step
#     mu1 = 10**(-5)  # parameter of the volume functional

#     # ----------------------------------------------------------------------
#     # -- Do not modify this cell, these are the values that you will be assessed against.
#     # ----------------------------------------------------------------------
#     # -- compute finite difference solution
#     u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
#                                    beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
#     chi0 = chi.copy()
#     u0 = u.copy()

#     # ----------------------------------------------------------------------
#     # -- Fell free to modify the function call in this cell.
#     # ----------------------------------------------------------------------
#     # -- compute optimization
#     # energy = numpy.zeros((100+1, 1), dtype=numpy.float64)
#     chi, energy, u, grad = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
#                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
#                        Alpha, mu, chi, V_obj,nb_pixels)
#     # --- en of optimization

#     chin = chi.copy()
#     un = u.copy()

#     # -- plot chi, u, and energy
#     postprocessing._plot_uncontroled_solution(u0, chi0)
#     postprocessing._plot_controled_solution(un, chin)
#     err = un - u0
#     postprocessing._plot_error(err)
#     postprocessing._plot_energy_history(energy)
#     matplotlib.pyplot.plot(energy)

#     matplotlib.pyplot.title('Energy')
#     matplotlib.pyplot.show()

#     print(energy[0], energy[-1])

#     print('End.')

# #plot_energy_of_frequencies(1)