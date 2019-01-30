# Linear Algebra 2018 Assignment 3

import numpy as np
import matplotlib.pyplot as plt
import math as m

# 6 points of a Octahedron(8 face)
# 稜長 = 根號2 
# center (0,0,0)
points = np.array([[ 0, 0, 1, 0, 0, -1],
                   [ 1, 0, 0, -1, 0, 0],
                   [ 0, 1, 0, 0, -1, 0]])


def plotcube(pt, name, hidden):
    """plot a cube described by pt. 
       T is the transition matrix that maps objects from a 3D space to a 2D screen.
       The viewport is at [1/2, 1/2, sqrt(2)/2]"""
    T = np.array([[m.sqrt(2)/m.sqrt(3), 0, -1/m.sqrt(3)],
                  [-1/m.sqrt(12),  m.sqrt(3)/2, -1/m.sqrt(6)]])
    
    def drawAxis():
        """ draw the axes of the 3D space"""
        X = np.dot(T, [[0,1.5],[0,0],[0,0]])
        Y = np.dot(T, [[0,0],[0,1.5],[0,0]])
        Z = np.dot(T, [[0,0],[0,0],[0,1.5]])
        plt.plot(X[0,:], X[1,:], 'b:')
        plt.plot(Y[0,:], Y[1,:], 'b:')
        plt.plot(Z[0,:], Z[1,:], 'b:')
        plt.text(X[0,1], X[1,1], r'x', fontsize=20)
        plt.text(Y[0,1], Y[1,1], r'y', fontsize=20)
        plt.text(Z[0,1]-0.1, Z[1,1], r'z', fontsize=20)

    def visible(p1, p2, p3):
        """output if the face is visible."""
        # write your code here...
        
        v = points[:, p2] - points[:, p1]      
        w = points[:, p3] - points[:, p1]
       
        normal = np.cross(w,v)

        view = np.array([[1/2],
                         [1/2],
                         [m.sqrt(2)/2]])
        num = np.dot(normal, view)       
        if num < 0:
            return True
        else:
            return False

    def mapRectangle(p1, p2, p3):
        """return two 1D arrays: X list and Y list from
           points[:, p1], points[:,p2], points[:, p3], points[:, p4]"""
        #### 1112: 只取p1,p2,p3,p4,p1這幾個column與T做dot
        A = np.dot(T, points[:, [p1,p2,p3,p1]]) #### 1112:給畫多邊形的點順序，可自己修改成你需要的N邊形
        return A[0,:], A[1,:]
    
    # plot face 1
    X1, Y1 = mapRectangle(0, 1, 2)
    if hidden == 0:
        plt.plot(X1,Y1)
    elif hidden == 1:
        if(visible(0, 1, 2)):
            plt.plot(X1,Y1)

    # plot face 2
    X2, Y2 = mapRectangle(0, 2, 4)
    if hidden == 0:
        plt.plot(X2,Y2)
    elif hidden == 1:
        if(visible(0, 2, 4)):
            plt.plot(X2,Y2)

    # plot face 3
    X3, Y3 = mapRectangle(0, 4, 5)
    if hidden == 0:
        plt.plot(X3,Y3)
    elif hidden == 1:
        if (visible(0, 4, 5)):
            plt.plot(X3,Y3)

    # plot face 4
    X4, Y4 = mapRectangle(0, 5, 1)
    if hidden == 0:
        plt.plot(X4,Y4)
    elif hidden == 1:
        if(visible(0, 5, 1)):
            plt.plot(X4,Y4)

    # plot face 5
    X5, Y5 = mapRectangle(3, 5, 4) 
    if hidden == 0:
        plt.plot(X5,Y5)
    elif hidden == 1:
        if(visible(3, 5, 4)):
            plt.plot(X5,Y5)

    # plot face 6
    X6, Y6 = mapRectangle(3, 1, 5)
    if hidden == 0:
        plt.plot(X6,Y6)
    elif hidden == 1:
        if(visible(3, 1, 5)):
            plt.plot(X6,Y6)

    # plot face 7
    X7, Y7 = mapRectangle(3, 2, 1)
    if hidden == 0:
        plt.plot(X7,Y7)
    elif hidden == 1:
        if (visible(3, 2, 1)):
            plt.plot(X7,Y7)

    # plot face 8
    X8, Y8 = mapRectangle(3, 4, 2)
    if hidden == 0:
        plt.plot(X8,Y8)
    elif hidden == 1:
        if(visible(3, 4, 2)):
            plt.plot(X8,Y8)

    plt.axis('equal')
    drawAxis()
    # store image
    filename = name + ".jpg"
    plt.savefig(filename)
    # change window title
    fig = plt.gcf()
    fig.canvas.set_window_title(name)
    # show figure on screen
    plt.show()
    
# ----------- the main body ----------------------

r1 = np.array([[m.sqrt(3)/2, 1/2, 0],
                   [-1/2, m.sqrt(3)/2, 0],
                   [0, 0, 1]])
  
r2 = np.array([[1, 0, 0.5],
               [0, 1, 0],
               [0, 0, 1]])

r3 = np.array([[4, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])

r4 = np.array([[1, 0, 0],
               [0, 1, 0],
               [2, 0, 1]])

#origin   
plotcube(points, "104070038_obj0", 0)
plotcube(points, "104070038_obj0_hid", 1)

#r1 旋轉30度
points = np.dot(r1, points)
plotcube(points, "104070038_obj1", 0)
plotcube(points, "104070038_obj1_hid", 1)

#r2 拉扯
points = np.array([[ 0, 0, 1, 0, 0, -1],
                   [ 1, 0, 0, -1, 0, 0],
                   [ 0, 1, 0, 0, -1, 0]])
points = np.dot(r2, points)
plotcube(points, "104070038_obj2", 0)
plotcube(points, "104070038_obj2_hid", 1)

#r3
points = np.array([[ 0, 0, 1, 0, 0, -1],
                   [ 1, 0, 0, -1, 0, 0],
                   [ 0, 1, 0, 0, -1, 0]])
points = np.dot(r3, points)
plotcube(points, "104070038_obj3", 0)
plotcube(points, "104070038_obj3_hid", 1)

#r4
points = np.array([[ 0, 0, 1, 0, 0, -1],
                   [ 1, 0, 0, -1, 0, 0],
                   [ 0, 1, 0, 0, -1, 0]])
points = np.dot(r4, points)
plotcube(points, "104070038_obj4", 0)
plotcube(points, "104070038_obj4_hid", 1)

# r1 r2
points = np.array([[ 0, 0, 1, 0, 0, -1],
                   [ 1, 0, 0, -1, 0, 0],
                   [ 0, 1, 0, 0, -1, 0]])
points = np.dot(r1, points)
points = np.dot(r2, points)
plotcube(points, "104070038_obj5", 0)
plotcube(points, "104070038_obj5_hid", 1)

# r1 r4
points = np.array([[ 0, 0, 1, 0, 0, -1],
                   [ 1, 0, 0, -1, 0, 0],
                   [ 0, 1, 0, 0, -1, 0]])
points = np.dot(r1, points)
points = np.dot(r4, points)
plotcube(points, "104070038_obj6", 0)
plotcube(points, "104070038_obj6_hid", 1)

# r1 r3
points = np.array([[ 0, 0, 1, 0, 0, -1],
                   [ 1, 0, 0, -1, 0, 0],
                   [ 0, 1, 0, 0, -1, 0]])
points = np.dot(r1, points)
points = np.dot(r3, points)
plotcube(points, "104070038_obj7", 0)
plotcube(points, "104070038_obj7_hid", 1)

# r2 r4
points = np.array([[ 0, 0, 1, 0, 0, -1],
                   [ 1, 0, 0, -1, 0, 0],
                   [ 0, 1, 0, 0, -1, 0]])
points = np.dot(r2, points)
points = np.dot(r4, points)
plotcube(points, "104070038_obj8", 0)
plotcube(points, "104070038_obj8_hid", 1)
