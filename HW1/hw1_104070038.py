# 2018 Linear Algebra assignment 1 example

import numpy as np
import matplotlib.pyplot as plt

# create the probability matrix of the graph
'''
A = np.array([[0, 0, 1, .5],
     [1.0/3, 0, 0, 0],
     [1.0/3,.5, 0, .5],
     [1.0/3,.5, 0,0]])
'''

A = np.array([[0, 0, 0, 0, 0, 0, 1.0/4, 0, 0, 1.0],
     [1.0/2, 0, 0, 0, 0, 1.0/2, 1.0/4, 0, 0, 0],
     [0, 1.0/3, 0, 1.0/2, 0, 0, 0, 0, 0, 0],
	 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	 [0, 1.0/3, 0, 0, 0, 0, 0, 0, 0, 0],
	 [0, 0, 0, 1.0/2, 1.0/3, 0, 0, 0, 1.0/2, 0],
	 [0, 0, 0, 0, 1.0/3, 0, 0, 0, 0, 0],
	 [1.0/2, 1.0/3, 0, 0, 1.0/3, 0, 1.0/4, 0, 0, 0],
	 [0, 0, 1.0/2, 0, 0, 1.0/2, 0, 0, 0, 0],
     [0, 0, 1.0/2, 0, 0, 0, 1.0/4, 1.0, 1.0/2, 0]])

# create an initial vector
'''
v = np.array([[0],[0],[0],[1]])
'''

v0 = np.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
v1 = np.array([[0],[1],[0],[0],[0],[0],[0],[0],[0],[0]])
v2 = np.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]])
v3 = np.array([[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]])
v4 = np.array([[0],[0],[0],[0],[1],[0],[0],[0],[0],[0]])
v5 = np.array([[0],[0],[0],[0],[0],[1],[0],[0],[0],[0]])
v6 = np.array([[0],[0],[0],[0],[0],[0],[1],[0],[0],[0]])
v7 = np.array([[0],[0],[0],[0],[0],[0],[0],[1],[0],[0]])
v8 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[1],[0]])
v9 = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]])


# compute their product
u0 = np.dot(A, v0)
u1 = np.dot(A, v1)
u2 = np.dot(A, v2)
u3 = np.dot(A, v3)
u4 = np.dot(A, v4)
u5 = np.dot(A, v5)
u6 = np.dot(A, v6)
u7 = np.dot(A, v7)
u8 = np.dot(A, v8)
u9 = np.dot(A, v9)

# define a function to compute |a-b|
#------------------------------------------------
def abssum(a, b):
    """this function computes the norm of a-b"""
    result = 0
    for i in range(len(a)):
        result = result + abs(a[i]-b[i])
    return result[0]
#------------------------------------------------

# diff records the difference of Av and v
diff_0 = [abssum(u0,v0)]
diff_1 = [abssum(u1,v1)]
diff_2 = [abssum(u2,v2)]
diff_3 = [abssum(u3,v3)]
diff_4 = [abssum(u4,v4)]
diff_5 = [abssum(u5,v5)]
diff_6 = [abssum(u6,v6)]
diff_7 = [abssum(u7,v7)]
diff_8 = [abssum(u8,v8)]
diff_9 = [abssum(u9,v9)]

i_0 = 0
i_1 = 0
i_2 = 0
i_3 = 0
i_4 = 0
i_5 = 0
i_6 = 0
i_7 = 0
i_8 = 0
i_9 = 0

while diff_0[i_0] > 1e-5:
    v0 = u0             # record A^kv
    u0 = np.dot(A, v0)  # compute A^{k+1}v
    diff_0.append(abssum(u0, v0))  
    i_0 = i_0 + 1

while diff_1[i_1] > 1e-5:
    v1 = u1             # record A^kv
    u1 = np.dot(A, v1)  # compute A^{k+1}v
    diff_1.append(abssum(u1, v1))  
    i_1 = i_1 + 1
	
while diff_2[i_2] > 1e-5:
    v2 = u2             # record A^kv
    u2 = np.dot(A, v2)  # compute A^{k+1}v
    diff_2.append(abssum(u2, v2))  
    i_2 = i_2 + 1	

while diff_3[i_3] > 1e-5:
    v3 = u3             # record A^kv
    u3 = np.dot(A, v3)  # compute A^{k+1}v
    diff_3.append(abssum(u3, v3))  
    i_3 = i_3 + 1

while diff_4[i_4] > 1e-5:
    v4 = u4             # record A^kv
    u4 = np.dot(A, v4)  # compute A^{k+1}v
    diff_4.append(abssum(u4, v4))  
    i_4 = i_4 + 1
	
while diff_5[i_5] > 1e-5:
    v5 = u5             # record A^kv
    u5 = np.dot(A, v5)  # compute A^{k+1}v
    diff_5.append(abssum(u5, v5))  
    i_5 = i_5 + 1
	
while diff_6[i_6] > 1e-5:
    v6 = u6             # record A^kv
    u6 = np.dot(A, v6)  # compute A^{k+1}v
    diff_6.append(abssum(u6, v6))  
    i_6 = i_6 + 1
	
while diff_7[i_7] > 1e-5:
    v7 = u7             # record A^kv
    u7 = np.dot(A, v7)  # compute A^{k+1}v
    diff_7.append(abssum(u7, v7))  
    i_7 = i_7 + 1
	
while diff_8[i_8] > 1e-5:
    v8 = u8             # record A^kv
    u8 = np.dot(A, v8)  # compute A^{k+1}v
    diff_8.append(abssum(u8, v8))  
    i_8 = i_8 + 1
	
while diff_9[i_9] > 1e-5:
    v9 = u9             # record A^kv
    u9 = np.dot(A, v9)  # compute A^{k+1}v
    diff_9.append(abssum(u9, v9))  
    i_9 = i_9 + 1	
	
print("v0: when k = " + str(i_0-1) + ", the difference will less than 1e-5")
print("v1: when k = " + str(i_1-1) + ", the difference will less than 1e-5")
print("v2: when k = " + str(i_2-1) + ", the difference will less than 1e-5")
print("v3: when k = " + str(i_3-1) + ", the difference will less than 1e-5")
print("v4: when k = " + str(i_4-1) + ", the difference will less than 1e-5")
print("v5: when k = " + str(i_5-1) + ", the difference will less than 1e-5")
print("v6: when k = " + str(i_6-1) + ", the difference will less than 1e-5")
print("v7: when k = " + str(i_7-1) + ", the difference will less than 1e-5")
print("v8: when k = " + str(i_8-1) + ", the difference will less than 1e-5")
print("v9: when k = " + str(i_9-1) + ", the difference will less than 1e-5")

print("probability matrix of v0~v9 when convergence:")
j = 0
print("v0: ")
while j<10:
	f = float(v0[j])
	print(f)
	j = j + 1

print("v1: ")
j = 0
while j<10:
	f = float(v1[j])
	print(f)
	j = j + 1

print("v2: ")
j = 0
while j<10:
	f = float(v2[j])
	print(f)
	j = j + 1

print("v3: ")
j = 0
while j<10:
	f = float(v3[j])
	print(f)
	j = j + 1
	
print("v4: ")
j = 0
while j<10:
	f = float(v4[j])
	print(f)
	j = j + 1

print("v5: ")
j = 0
while j<10:
	f = float(v5[j])
	print(f)
	j = j + 1	
	
print("v6: ")
j = 0
while j<10:
	f = float(v6[j])
	print(f)
	j = j + 1
	
print("v7: ")
j = 0
while j<10:
	f = float(v7[j])
	print(f)
	j = j + 1
	
print("v8: ")
j = 0
while j<10:
	f = float(v8[j])
	print(f)
	j = j + 1
	
print("v9: ")
j = 0
while j<10:
	f = float(v9[j])
	print(f)
	j = j + 1
'''
while j<10:
	f = float(v0[j])
	sum = sum + f
	j = j + 1
	
	if j == 10:
		if sum != 1:
			sum = sum - f
			f = f + abs(1-sum)
			sum = sum + f
	
	print(f)
	
print("sum of ten probabbilities:" + str(sum))
print(abs(1-sum) + sum)
'''
	
#plot the differences with iterations
plt.semilogy(range(len(diff_0)), diff_0)
plt.semilogy(range(len(diff_1)), diff_1)
plt.semilogy(range(len(diff_2)), diff_2)
plt.semilogy(range(len(diff_3)), diff_3)
plt.semilogy(range(len(diff_4)), diff_4)
plt.semilogy(range(len(diff_5)), diff_5)
plt.semilogy(range(len(diff_6)), diff_6)
plt.semilogy(range(len(diff_7)), diff_7)
plt.semilogy(range(len(diff_8)), diff_8)
plt.semilogy(range(len(diff_9)), diff_9)


plt.xlabel('iterations')
plt.ylabel('difference')
plt.title('Convergence')
plt.show()
