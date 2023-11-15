import numpy
import _env
import math
def compute_gradient_descent_moghit(chi, grad, domain, mu):
    (M, N) = numpy.shape(domain)
    for i in range(1,M-1):
        for j in range(1,N-1):
            L=[]
            if domain[i,j]==_env.NODE_ROBIN:
                if domain[i+1,j]!=_env.NODE_COMPLEMENTARY: #and grad[i+1,j]!=0:
                    L.append(grad[i+1,j])
                elif domain[i,j+1]!=_env.NODE_COMPLEMENTARY :#and grad[i,j+1]!=0:
                    L.append(grad[i,j+1])
                elif domain[i,j-1]!=_env.NODE_COMPLEMENTARY: #and grad[i,j-1]!=0:  
                    L.append(grad[i,j-1])
                elif domain[i-1,j]!=_env.NODE_COMPLEMENTARY: #and grad[i-1,j]!=0:
                    L.append(grad[i-1,j])
                
                if len(L)!=0:
                    max1=max(L)
                    min1=min(L)
                    if abs(max1)>abs(min1):
                        grade=max1
                    else:
                        grade=min1
                    chi[i,j] = chi[i,j] - mu*grade
                else:
                    print((i,j))
            elif domain[i,j]!=_env.NODE_COMPLEMENTARY:
                chi[i,j] = chi[i,j] -mu*grad[i,j]
    return chi
