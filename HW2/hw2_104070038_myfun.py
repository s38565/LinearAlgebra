import numpy as np
from datetime import datetime
#---------------------------------------------------------------
def mydet(A):
    """"compute deteminant of A using cofactor expansion."""
    n = A.shape[0]
    if n == 1:
        return A[0][0]
    elif n == 2:
        return A[0][0]*A[1][1]-A[0][1]*A[1][0]
    else:
        det = 0
        for i in range(n):
            # compute minor M(0,i)
            ### you can use np.concatenate((A,B), axis=1) to merge two matrices
            if i==0:
                M = A[1:n,1:n]
            else:
                X = A[1:n,0:i]
                Y = A[1:n,i+1:n]
                M = np.concatenate((X,Y), axis=1)
            # compute cofactor A(0,i) = (-1)^{i} det(M(0,i))
            
            if i%2==0:
                C = mydet(M)
            else:
                C = -mydet(M)
            det = det + A[0][i] * C
            ###   call mydet recursively to compute det(M)
            
        return det
#---------------------------------------------------------------
def mysolve_adj(A, b, detA):
    # TODO 1. solve Ax=b using adjoint matrix (using mydet)
    n = A.shape[0]
    ans = np.empty((n,1))
    # M = inverse matrix of A
    if n == 2:
        adj = [[A[1][1]/detA, -1*A[0][1]/detA],
                [-1*A[1][0]/detA, A[0][0]/detA]]
    else:
        M = np.empty((n-1,n-1))
        adj = np.empty((n,n))
        i = 0
        
        while i<n:
            j = 0
            while j<n:
                if i==0:
                    X = A[1:n,0:j]
                    Y = A[1:n,j+1:n]
                    M = np.concatenate((X,Y), axis=1)
                elif i<n-1:
                    X = A[0:i,0:j]
                    Y = A[0:i,j+1:n]
                    N = np.concatenate((X,Y), axis=1)
                    U = A[i+1:n,0:j]
                    V = A[i+1:n,j+1:n]
                    P = np.concatenate((U,V), axis=1)
                    M = np.concatenate((N,P), axis=0)
                elif i == n-1:
                    X = A[0:n-1,0:j]
                    Y = A[0:n-1,j+1:n]
                    M = np.concatenate((X,Y), axis=1)
                sign = (-1)**(i+j)
                adj[j][i] = sign * mydet(M) / detA
                j = j + 1
            i = i + 1
    # x = A^-1 * b
    ans = np.dot(adj, b)
    return ans
#---------------------------------------------------------------
def mysolve_cramer(A, b, detA):
    # TODO 2. solve Ax=b using Cramer's rule (using mydet)
    # return ?
    n = A.shape[0]
    ans = np.empty((n,1))
    for i in range(n):
        if i==0:
            X = A[0:n,1:n]
            M = np.concatenate((b,X), axis=1)
            ans[i] = mydet(M)/detA
        else:
            X = A[0:n,0:i]
            Z = np.concatenate((X,b), axis=1)
            Y = A[0:n,i+1:n]
            M = np.concatenate((Z,Y), axis=1)
            ans[i] = mydet(M)/detA

    return ans
#---------------------------------------------------------------
'''
A = np.array([[3,4,-3],
              [3,-2,4],
              [3,2,-1]])

b = np.array([[5],[7],[3]])
print(mysolve_cramer(A, b, mydet(A)))
print(mysolve_adj(A, b, mydet(A))) 
'''