# -*- coding: utf-8 -*-


# Python packages
import postprocessing
import processing
import preprocessing
import _env
import matplotlib.pyplot
import numpy
import os
import sys
import descente_grad
numpy.set_printoptions(threshold=sys.maxsize)

# fonctions utilitaires


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

    (M, N) = numpy.shape(domain_omega)
    spacestep_x = 1.0 / N
    spacestep_y = 1.0 / M

    energy = 0.0
    # energy = numpy.sum(numpy.sum(numpy.abs(u)**2))

    for i in range(0, M-1):
        for j in range(0, N-1):
            if domain_omega[i, j] == _env.NODE_INTERIOR or domain_omega[i+1, j] == _env.NODE_INTERIOR or domain_omega[i, j+1] == _env.NODE_INTERIOR or domain_omega[i+1, j+1] == _env.NODE_INTERIOR:
                energy += (numpy.abs(u[i, j])**2 + numpy.abs(u[i+1, j])**2 +
                           # ou spacestep_x*spacestep_x/4
                           numpy.abs(u[i, j+1])**2 + numpy.abs(u[i+1, j+1])**2)*spacestep_x*spacestep_x/4

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


def final_projected(chi, nb_pixels):
    # find the beta th largest value of chi
    chi_copy = chi.copy()
    chi_copy = chi_copy.reshape(-1)
    chi_copy.sort()
    chi_copy = chi_copy[::-1]
    threshold = chi_copy[nb_pixels-1]
    # set to zero all the values of chi that are smaller than the threshold
    chi[chi < threshold] = 0.
    # set to one all the values of chi that are greater than the threshold
    chi[chi >= threshold] = 1.
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
    # chi[50,:] = chi[50,:] - mu*grad[50,:]
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            # print(i,j)
            # chi[i,j] = chi[i,j] - mu * grad[i,j]
            a = BelongsInteriorDomain(domain[i + 1, j])
            b = BelongsInteriorDomain(domain[i - 1, j])
            c = BelongsInteriorDomain(domain[i, j + 1])
            d = BelongsInteriorDomain(domain[i, j - 1])
            if a == 2:
                # print(i+1, j, "-----", "i+1,j")
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

# algo d'optimisation pour un seul k


def your_optimization_procedure(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                Alpha, mu, chi, V_obj, beta):
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
    chi_list = numpy.zeros((numb_iter+1, M, N), dtype=numpy.float64)
    chi_list[0, :, :] = chi
    while k < numb_iter and mu > 10**(-5):
        print('---- iteration number = ', k)
        # print('1. computing solution of Helmholtz problem, i.e., u')
        chi = chi_list[k, :, :]
        alpha_rob = Alpha*chi
        u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        # print('2. computing solution of adjoint problem, i.e., p')
        q = processing.solve_helmholtz(domain_omega, spacestep, omega, -2*numpy.conjugate(u), g_dir, g_neu, g_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        # print('3. computing objective function, i.e., energy')
        energy[k] = your_compute_objective_function(domain_omega, u, spacestep)
        ene = energy[k, 0]
        # print('4. computing parametric gradient')
        grad = numpy.real(Alpha*u*q)
        while ene >= energy[k, 0] and mu > 10**(-5):
            # print('    a. computing gradient descent')
            old_chi = chi_list[k, :, :].copy()
            new_chi = compute_gradient_descent(
                old_chi, grad, domain_omega, mu)
            # print('    b. computing projected gradient')
            new_chi = compute_projected(new_chi, domain_omega, V_obj)
            new_chi = processing.set2zero(new_chi, domain_omega)
            # print('    c. computing solution of Helmholtz problem, i.e., u')
            alpha_rob = Alpha*new_chi
            u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            # print('    d. computing objective function, i.e., energy (E)')
            ene = your_compute_objective_function(domain_omega, u, spacestep)
            if ene < energy[k]:
                # The step is increased if the energy decreased
                mu = mu * 1.1
                # print('sortie')
                chi_list[k+1, :, :] = new_chi.copy()
            else:
                # The step is decreased if the energy increased
                mu = mu / 2
            print("mu=" + str(mu), "ene=" + str(ene),
                  "grad=" + str(numpy.linalg.norm(grad, 2)))
        k += 1
    chi = chi_list[k-1, :, :]
    chi = final_projected(chi, beta)
    alpha_rob = Alpha*chi
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    energy = energy[:k]
    energy[-1] = your_compute_objective_function(domain_omega, u, spacestep)
    return chi, energy, u  # , grad


def simulated_annealing(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                        Alpha, mu, chi, V_obj, beta):
    numb_iter = 20
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    (M, N) = numpy.shape(domain_omega)
    # initial point
    best = chi.copy()
    alpha_rob = Alpha*chi
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    best_val = your_compute_objective_function(domain_omega, u, spacestep)
    energy[0] = best_val
    curr, curr_eval = best, best_val
    # initial temperature
    T = 100
    for k in range(numb_iter):
        candidate = curr+numpy.random.uniform(0, 1, size=(M, N))
        candidate = compute_projected(candidate, domain_omega, V_obj)
        alpha_rob = Alpha*candidate
        u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        candidate_val = your_compute_objective_function(
            domain_omega, u, spacestep)
        energy[k+1] = candidate_val
        if candidate_val < best_val:
            best, best_val = candidate, candidate_val
            print('>%d f=%f' % (k, best_val))
        diff = candidate_val - curr_eval
        if diff < 0 or numpy.exp(-diff / T) > numpy.random.uniform(0, 1):
            curr, curr_eval = candidate, candidate_val
        T = T * 0.99
    chi = final_projected(best, beta)
    alpha_rob = Alpha*chi
    u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    energy[-1] = your_compute_objective_function(domain_omega, u, spacestep)
    return chi, energy, u


def random_generated(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                     beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                     Alpha, mu, chi, V_obj, beta):
    energy = numpy.zeros((2, 1), dtype=numpy.float64)
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    energy[0, 0] = your_compute_objective_function(domain_omega, u, spacestep)
    chi = numpy.random.uniform(0, 1, size=(M, N))
    chi = compute_projected(chi, domain_omega, V_obj)
    chi = final_projected(chi, beta)
    alpha_rob = Alpha*chi
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    energy[1, 0] = your_compute_objective_function(domain_omega, u, spacestep)
    print("energy=" + str(energy))
    return chi, energy, u

# algo d'optimisation pour plusieurs k


def opti_multi_freq_gradient(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                             beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                             Alpha, mu, chi, V_obj, beta):

    wavenumbers = numpy.array([k for k in range(1, 100)])
    # on calcule l'energie pour chaque k pour trouver les coefficients de ponderation
    energy_coef = numpy.zeros(wavenumbers.shape)
    alpha_rob = Alpha*chi
    for j in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[j], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energy_coef[j] = energy_coef[j] + \
            your_compute_objective_function(domain_omega, u, spacestep)
    K = 10
    # on trouver les K plus grandes valeurs de l'energie et les indices correspondants
    indices = numpy.argsort(energy_coef, axis=0)[::-1][:K]
    wavenumbers = wavenumbers[indices]
    energy_coef = energy_coef[indices]

    g_dir = numpy.zeros(f_dir.shape)
    g_neu = numpy.zeros(f_neu.shape)
    g_rob = numpy.zeros(f_rob.shape)
    k = 0
    (M, N) = numpy.shape(domain_omega)
    numb_iter = 10
    energy = numpy.zeros((numb_iter+1, 1), dtype=numpy.float64)
    chi_list = numpy.zeros((numb_iter+1, M, N), dtype=numpy.float64)
    chi_list[0, :, :] = chi
    while k < numb_iter and mu > 10**(-5):
        print('---- iteration number = ', k)
        grad = 0
        chi = chi_list[k, :, :]
        alpha_rob = Alpha*chi
        for j in range(len(wavenumbers)):
            # print('1. computing solution of Helmholtz problem, i.e., u')
            u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[j], f, f_dir, f_neu, f_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            # print('2. computing solution of adjoint problem, i.e., p')
            q = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[j], -2*numpy.conjugate(u), g_dir, g_neu, g_rob,
                                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            # print('3. computing objective function, i.e., energy')
            energy[k] = energy[k] + your_compute_objective_function(
                domain_omega, u, spacestep)*energy_coef[j]
            # print('4. computing parametric gradient')
            grad = grad + numpy.real(Alpha*u*q)*energy_coef[j]
        ene = energy[k, 0]
        while ene >= energy[k, 0] and mu > 10**(-5):
            # print('    a. computing gradient descent')
            old_chi = chi_list[k, :, :].copy()
            new_chi = compute_gradient_descent(
                old_chi, grad, domain_omega, mu)
            # print('    b. computing projected gradient')
            new_chi = compute_projected(new_chi, domain_omega, V_obj)
            new_chi = processing.set2zero(new_chi, domain_omega)
            # print('    c. computing solution of Helmholtz problem, i.e., u')
            alpha_rob = Alpha*new_chi
            ene = 0
            for j in range(len(wavenumbers)):
                u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[j], f, f_dir, f_neu, f_rob,
                                               beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
            # print('    d. computing objective function, i.e., energy (E)')
                ene += your_compute_objective_function(
                    domain_omega, u, spacestep)*energy_coef[j]
            if ene < energy[k]:
                # The step is increased if the energy decreased
                mu = mu * 1.1
                # print('sortie')
                chi_list[k+1, :, :] = new_chi.copy()
            else:
                # The step is decreased if the energy increased
                mu = mu / 2
            print("mu=" + str(mu), "ene=" + str(ene),
                  "grad=" + str(numpy.linalg.norm(grad, 2)))
        k += 1
    chi = chi_list[k-1, :, :]
    chi = final_projected(chi, beta)
    # alpha_rob = Alpha*chi
    # u = processing.solve_helmholtz(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
    #                                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    energy = energy[:k]
    # energy[-1] = your_compute_objective_function(domain_omega, u, spacestep)
    return chi, energy, u


def opti_multi_freq_random(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj, beta):

    wavenumbers = numpy.array([k for k in range(1, 100)])
    energies_before = numpy.zeros((len(wavenumbers), 1))
    energies = numpy.zeros((len(wavenumbers), 1))
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ene = your_compute_objective_function(domain_omega, u, spacestep)
        energies_before[k] = ene
    chi, _, _ = random_generated(domain_omega, spacestep, wavenumbers[0], f, f_dir, f_neu, f_rob,
                                 beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                 Alpha, mu, chi, V_obj, beta)
    alpha_rob = Alpha*chi
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ene = your_compute_objective_function(domain_omega, u, spacestep)
        energies[k] = ene
    # postprocessing._plot_controled_solution(u, chi)
    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies_before, label='Avant optimisation')
    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies, label='Après optimisation')
    matplotlib.pyplot.xlabel('Fréquence')
    matplotlib.pyplot.ylabel('Energie')
    # limit to 3 digits in title
    print('Energie maximale avant optimisation = ' +
          str(numpy.round(numpy.max(energies_before), 4))+', Moy = '+str(numpy.round(numpy.average(energies_before), 3)))
    matplotlib.pyplot.title('Energie maximale après optimisation = ' +
                            str(numpy.round(numpy.max(energies), 4))+', Moy = '+str(numpy.round(numpy.average(energies), 3)))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    return chi


# plot l'energie en fonction de la frequence pour un chi donné en entrée


def energy_freq(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                Alpha, mu, chi, V_obj, beta):
    wavenumbers = numpy.array([k for k in range(1, 100)])
    energies = numpy.zeros((len(wavenumbers), 1))
    alpha_rob = Alpha*chi
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energies[k] = your_compute_objective_function(
            domain_omega, u, spacestep)

    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies)
    matplotlib.pyplot.xlabel('Fréquence')
    matplotlib.pyplot.ylabel('Energie')
    # limit to 3 digits in title
    matplotlib.pyplot.title(' Max = ' +
                            str(numpy.round(numpy.max(energies), 3))+', Moy = '+str(numpy.round(numpy.average(energies), 3)))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    return

# plot l'energie en fonction de la frequence pour un chi donné en entrée et pour un chi totalement absorbant


def energy_freq_vs_fullabs(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj, beta):
    wavenumbers = numpy.array([k for k in range(1, 100)])
    energies = numpy.zeros((len(wavenumbers), 1))
    alpha_rob = Alpha*chi
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energies[k] = your_compute_objective_function(
            domain_omega, u, spacestep)
    # full absorbing chi
    energies_fullabs = numpy.zeros((len(wavenumbers), 1))
    chi = preprocessing._set_chi(M, N, x, y)
    chi = numpy.ones(chi.shape)
    chi = preprocessing.set2zero(chi, domain_omega)
    alpha_rob = Alpha*chi
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energies_fullabs[k] = your_compute_objective_function(
            domain_omega, u, spacestep)

    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies, label='chi optimisé')
    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies_fullabs, label='chi totalement absorbant')
    matplotlib.pyplot.xlabel('Fréquence')
    matplotlib.pyplot.ylabel('Energie')
    # limit to 3 digits in title
    matplotlib.pyplot.title(' Max = ' +
                            str(numpy.round(numpy.max(energies), 3))+', Moy = '+str(numpy.round(numpy.average(energies), 3)))
    print('Energie max pour chi totalement absorbant = ' + str(numpy.round(numpy.max(energies_fullabs), 3)) + ', Moy = ' + str(
        numpy.round(numpy.average(energies_fullabs), 3)))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    return

# plot l'energie en fonction de la frequence pour un chi donné en entrée et pour un chi initial non optimisé


def energy_freq_vs_non_optimised(domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
                                 beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                 Alpha, mu, chi, V_obj, beta):
    wavenumbers = numpy.array([k for k in range(1, 100)])
    energies = numpy.zeros((len(wavenumbers), 1))
    alpha_rob = Alpha*chi
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energies[k] = your_compute_objective_function(
            domain_omega, u, spacestep)
    # full absorbing chi
    energies_non_op = numpy.zeros((len(wavenumbers), 1))
    chi = preprocessing._set_chi(M, N, x, y)
    chi = preprocessing.set2zero(chi, domain_omega)
    alpha_rob = Alpha*chi
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        energies_non_op[k] = your_compute_objective_function(
            domain_omega, u, spacestep)
    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies_non_op, label='chi non optimisé')
    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies, label='chi optimisé')

    matplotlib.pyplot.xlabel('Fréquence')
    matplotlib.pyplot.ylabel('Energie')
    # limit to 3 digits in title
    matplotlib.pyplot.title(' Max = ' +
                            str(numpy.round(numpy.max(energies), 3))+', Moy = '+str(numpy.round(numpy.average(energies), 3)))
    print('Energie max pour chi non op  = ' + str(numpy.round(numpy.max(energies_non_op), 3)) + ', Moy = ' + str(
        numpy.round(numpy.average(energies_non_op, 3))))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()
    return


# plot l'energie en fonction de la frequence pour 2 chi différents donnés en entrée

def plot2chi(chi1, chi2, domain_omega, spacestep, omega, f, f_dir, f_neu, f_rob,
             beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
             Alpha, mu, chi, V_obj, beta):
    wavenumbers = numpy.array([k for k in range(1, 100)])
    energies1 = numpy.zeros((len(wavenumbers), 1))
    energies2 = numpy.zeros((len(wavenumbers), 1))
    alpha_rob = Alpha*chi1
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ene = your_compute_objective_function(domain_omega, u, spacestep)
        energies1[k] = ene
    chi2, _, _ = random_generated(domain_omega, spacestep, wavenumbers[0], f, f_dir, f_neu, f_rob,
                                  beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                  Alpha, mu, chi, V_obj, beta)
    alpha_rob = Alpha*chi2
    for k in range(len(wavenumbers)):
        u = processing.solve_helmholtz(domain_omega, spacestep, wavenumbers[k], f, f_dir, f_neu, f_rob,
                                       beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
        ene = your_compute_objective_function(domain_omega, u, spacestep)
        energies2[k] = ene
    postprocessing._plot_controled_solution(u, chi)
    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies1, label='descente de gradient')
    matplotlib.pyplot.plot(wavenumbers*(340/(2*numpy.pi)),
                           energies2, label='aléatoire')
    matplotlib.pyplot.xlabel('Fréquence')
    matplotlib.pyplot.ylabel('Energie')
    # limit to 3 digits in title
    matplotlib.pyplot.title(' Max = ' +
                            str(numpy.round(numpy.max(energies2), 3))+', Moy = '+str(numpy.round(numpy.average(energies2), 3)))
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


if __name__ == '__main__':

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- set parameters of the geometry
    N = 100  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 2
    spacestep = 1.0 / N  # mesh size

    # -- set parameters of the partial differential equation
    kx = -1.0
    ky = -1.0
    wavenumber = numpy.sqrt(kx**2 + ky**2)  # wavenumber
    wavenumber = 10.

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # --- set coefficients of the partial differential equation
    beta_pde, alpha_pde, alpha_dir, beta_neu, alpha_rob, beta_rob = preprocessing._set_coefficients_of_pde(
        M, N)

    # -- set right hand sides of the partial differential equation
    f, f_dir, f_neu, f_rob = preprocessing._set_rhs_of_pde(M, N)

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(
        M, N, level)

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- define boundary conditions
    # planar wave defined on top
    f_dir[:, :] = 0.0
    f_dir[0, 0:N] = 1.0
    # spherical wave defined on top
    # f_dir[:, :] = 0.0
    # f_dir[0, int(N/2)] = 10.0

    # -- initialize
    alpha_rob[:, :] = - wavenumber * 1j

    # -- define material density matrix
    chi = preprocessing._set_chi(M, N, x, y)
    # chi = numpy.ones(chi.shape)
    chi = preprocessing.set2zero(chi, domain_omega)
    nb_pixels = int(numpy.sum(numpy.sum(chi)))
    # -- define absorbing material
    Alpha = 10.0 - 10.0 * 1j
    # -- this is the function you have written during your project
    # import compute_alpha
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
    # print('V_obj=' + str(V_obj))

    mu = 5  # initial gradient step
    mu1 = 10**(-5)  # parameter of the volume functional

    # ----------------------------------------------------------------------
    # -- Do not modify this cell, these are the values that you will be assessed against.
    # ----------------------------------------------------------------------
    # -- compute finite difference solution
    u = processing.solve_helmholtz(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob)
    chi0 = chi.copy()
    u0 = u.copy()

    # ----------------------------------------------------------------------
    # -- Fell free to modify the function call in this cell.
    # ----------------------------------------------------------------------
    # -- compute optimization to find chi

    # chi, energy, u = your_optimization_procedure(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                                              beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                                              Alpha, mu, chi, V_obj, nb_pixels)

    # chi, energy, u = simulated_annealing(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                                      beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                                      Alpha, mu, chi, V_obj, nb_pixels)

    # chi, energy, u = random_generated(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                                   beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                                   Alpha, mu, chi, V_obj, nb_pixels)
    chi, energy, u = opti_multi_freq_gradient(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                                              beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                                              Alpha, mu, chi, V_obj, nb_pixels)
    # --- end of optimization
    #
    # -- plot energy with the found during the optimization procedure
    energy_freq_vs_fullabs(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
                           beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
                           Alpha, mu, chi, V_obj, nb_pixels)

    # energy_freq_vs_non_optimised(domain_omega, spacestep, wavenumber, f, f_dir, f_neu, f_rob,
    #                        beta_pde, alpha_pde, alpha_dir, beta_neu, beta_rob, alpha_rob,
    #                        Alpha, mu, chi, V_obj, nb_pixels)
    # -- end of plotting energy
    chin = chi.copy()
    un = u.copy()

    # -- plot chi, u, and energy
    postprocessing._plot_uncontroled_solution(u0, chi0)
    postprocessing._plot_controled_solution(un, chin)
    err = un - u0
    postprocessing._plot_error(err)
    postprocessing._plot_energy_history(energy)
    print("Energie initial : " +
          str(energy[0, 0]) + ", Energie optimisée : " + str(energy[-1, 0]))
    print('End.')
