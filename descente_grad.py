import numpy
import _env


def compute_gradient_descent(chi, grad, domain, mu):
    (M, N) = numpy.shape(domain)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if domain[i, j] == _env.NODE_ROBIN:
                grade = 0
                c = 0
                if domain[i+1, j] == _env.NODE_INTERIOR:
                    grade += grad[i+1, j]
                    c += 1
                if domain[i, j+1] == _env.NODE_INTERIOR:
                    grade += grad[i, j+1]
                    c += 1
                if domain[i, j-1] == _env.NODE_INTERIOR:
                    grade = grade + grad[i, j-1]
                    c += 1
                if domain[i-1, j] == _env.NODE_INTERIOR:
                    grade += grad[i-1, j]
                    c += 1
                if c != 0:
                    grade = grade/c
                else:
                    grade = 0
                grade = grade/4
                chi[i, j] = chi[i, j] - mu*grade
            # elif domain[i,j]==_env.NODE_INTERIOR:
            #     chi[i,j] = chi[i,j] -mu*grad[i,j]
    return chi
# import numpy
# import _env


def compute_gradient_descent2(chi, grad, domain, mu):
    (M, N) = numpy.shape(domain)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if domain[i, j] == _env.NODE_ROBIN:
                grade = 0
                if domain[i+1, j] == _env.NODE_INTERIOR:
                    grade += grad[i+1, j]
                elif domain[i, j+1] == _env.NODE_INTERIOR:
                    grade += grad[i, j+1]
                elif domain[i, j-1] == _env.NODE_INTERIOR:
                    grade = grade + grad[i, j-1]
                elif domain[i-1, j] == _env.NODE_INTERIOR:
                    grade += grad[i-1, j]
                grade = grade/4
                chi[i, j] = chi[i, j] - mu*grade
            elif domain[i, j] == _env.NODE_INTERIOR:
                chi[i, j] = chi[i, j] - mu*grad[i, j]
    return chi
