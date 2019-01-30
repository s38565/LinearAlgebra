#Linear Algbra 2018 Assignment 2
from hw2_104070038_myfun import mydet, mysolve_cramer, mysolve_adj

import numpy as np
from datetime import datetime

A = np.array([[1,-1,-1,0,0,0,0,0,0,0,0],
              [0,0,1,-1,-1,0,0,0,0,0,0],
              [0,0,0,0,1,-1,-1,0,0,0,0],
              [0,0,0,0,0,0,1,-1,-1,0,0],
              [0,0,0,0,0,0,0,0,1,-1,-1],
              [3,3,0,0,0,0,0,0,0,0,0],
              [0,3,-4,-2,0,0,0,0,0,0,0],
              [0,0,0,2,-1,-5,0,0,0,0,0],
              [0,0,0,0,0,5,-2,-4,0,0,0],
              [0,0,0,0,0,0,0,4,-3,-3,0],
              [0,0,0,0,0,0,0,0,0,3,-2]])

b = np.array([[0],[0],[0],[0],[0],[15],[0],[0],[0],[0],[10]])
n = A.shape[0]


# compute the determinant of A using numpy
print("determinant using numpy: ")
tStart = datetime.now()
print(np.linalg.det(A))
tEnd = datetime.now()
print("Time to solve determinent using np.linalg.det is ", tEnd-tStart, " seconds.\n")

# compute the determinant of A using mydet
### your code write in hw2_StudentID_myfun.py ("mydet" function) ###
print("determinant using mydet: ")
tStart = datetime.now()
print(mydet(A))
tEnd = datetime.now()
print("Time to solve determinent using mydet is ", tEnd-tStart, " seconds.\n")

# solve the linear system and measure the execution time
tStart = datetime.now()
x = np.linalg.solve(A, b)
tEnd = datetime.now()
print("Time to solve Ax=b using np.linalg.solve is ", tEnd-tStart, " seconds.")

# check the correctness
print("rediduals of Ax=b using np.linalg.solve: ")
res = np.subtract(np.dot(A, x), b)
print(res,"\n")


# TODO 1. solve Ax=b using adjoint matrix (using mydet)
### your code write in hw2_StudentID_myfun.py ("mysolve_adj" function) ###
# execution time
tStart = datetime.now()
x1 = mysolve_adj(A, b, mydet(A))
tEnd = datetime.now()
print("Execution Time using adjoint matrix = ", tEnd-tStart, " seconds.\n")

#rediduals
print("rediduals of Ax=b using adjoint matrix: ")
rediduals = np.subtract(np.dot(A, x1), b)
print(rediduals,"\n")



# TODO 2. solve Ax=b using Cramer's rule (using mydet)
### your code write in hw2_StudentID_myfun.py ("mysolve_cramer" function) ###
# execution time
tStart = datetime.now()
x2 = mysolve_cramer(A, b, mydet(A))
tEnd = datetime.now()
print("Execution Time using Cramer's rule = ", tEnd-tStart, " seconds.\n")

#rediduals
print("rediduals of Ax=b using Cramer's rule: ")
rediduals = np.subtract(np.dot(A, x2), b)
print(rediduals)
