import numpy
import _env
import math


def compute_gradient_descent_moghit(chi, grad, domain, mu):
    (M, N) = numpy.shape(domain)
    for i in range(1, M-1):
        for j in range(1, N-1):
            L = []
            if domain[i, j] == _env.NODE_ROBIN:
                # and grad[i+1,j]!=0:
                if domain[i+1, j] != _env.NODE_COMPLEMENTARY:
                    L.append(grad[i+1, j])
                # and grad[i,j+1]!=0:
                elif domain[i, j+1] != _env.NODE_COMPLEMENTARY:
                    L.append(grad[i, j+1])
                # and grad[i,j-1]!=0:
                elif domain[i, j-1] != _env.NODE_COMPLEMENTARY:
                    L.append(grad[i, j-1])
                # and grad[i-1,j]!=0:
                elif domain[i-1, j] != _env.NODE_COMPLEMENTARY:
                    L.append(grad[i-1, j])

                if len(L) != 0:
                    max1 = max(L)
                    min1 = min(L)
                    if abs(max1) > abs(min1):
                        grade = max1
                    else:
                        grade = min1
                    chi[i, j] = chi[i, j] - mu*grade
                else:
                    print((i, j))
            elif domain[i, j] != _env.NODE_COMPLEMENTARY:
                chi[i, j] = chi[i, j] - mu*grad[i, j]
    return chi


def compute_gradient_descent(chi, grad, domain, mu):
    (M, N) = numpy.shape(domain)
    for i in range(1, M-1):
        for j in range(1, N-1):
            if domain[i, j] == _env.NODE_ROBIN:
                grade = 0
                # c = 0
                if domain[i+1, j] == _env.NODE_INTERIOR:
                    grade += grad[i+1, j]
                    # c += 1
                if domain[i, j+1] == _env.NODE_INTERIOR:
                    grade += grad[i, j+1]
                    # c += 1
                if domain[i, j-1] == _env.NODE_INTERIOR:
                    grade = grade + grad[i, j-1]
                    # c += 1
                if domain[i-1, j] == _env.NODE_INTERIOR:
                    grade += grad[i-1, j]
                    # c += 1

                grade = grade/4
                # else:
                #     grade = 0
                chi[i, j] = chi[i, j] - mu*grade
            # elif domain[i,j]==_env.NODE_INTERIOR:
            #     chi[i,j] = chi[i,j] -mu*grad[i,j]
    return chi
