from sys import path
import numpy as np
from math import cos,sin,sqrt
path.append(r"/home/robot/workspaces/casadi")
from casadi import *

def get_cpose( q1,  q2,  q3,  q4,  q5,  q6):
    T = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    z_sh = 0.1
    T[0][0] = 0.06*sin(q1)
    T[0][1] = -0.06*cos(q1)
    T[0][2] = 0.0894+z_sh
    T[1][0] = (-0.425*cos(q1)*cos(q2))/2+0.14*sin(q1)
    T[1][1] = (-0.425*cos(q2)*sin(q1))/2-0.14*cos(q1)
    T[1][2] = (0.0894 - 0.425*sin(q2))/2+z_sh
    T[2][0] = -0.425*cos(q1)*cos(q2)+0.11*sin(q1)
    T[2][1] = -0.425*cos(q2)*sin(q1)-0.11*cos(q1)
    T[2][2] = 0.0894 - 0.425*sin(q2)+z_sh
    T[3][0] = -0.425*cos(q1)*cos(q2)+(-(cos(q1)*(1569.0*cos(q2 + q3) + 1700.0*cos(q2)))/4000-(-0.425*cos(q1)*cos(q2)))/3+0.02*sin(q1)
    T[3][1] = -0.425*cos(q2)*sin(q1)+(-(sin(q1)*(1569.0*cos(q2 + q3) + 1700.0*cos(q2)))/4000-(-0.425*cos(q2)*sin(q1)))/3-0.02*cos(q1)
    T[3][2] = 0.0894 - 0.425*sin(q2)+(0.0894 - 0.425*sin(q2) - 0.39225*sin(q2 + q3)-(0.0894 - 0.425*sin(q2)))/3+z_sh
    T[4][0] = -0.425*cos(q1)*cos(q2)+2*(-(cos(q1)*(1569.0*cos(q2 + q3) + 1700.0*cos(q2)))/4000-(-0.425*cos(q1)*cos(q2)))/3+0.02*sin(q1)
    T[4][1] = -0.425*cos(q2)*sin(q1)+2*(-(sin(q1)*(1569.0*cos(q2 + q3) + 1700.0*cos(q2)))/4000-(-0.425*cos(q2)*sin(q1)))/3-0.02*cos(q1)
    T[4][2] = 0.0894 - 0.425*sin(q2)+2*(0.0894 - 0.425*sin(q2) - 0.39225*sin(q2 + q3)-(0.0894 - 0.425*sin(q2)))/3+z_sh
    T[5][0] = -(cos(q1)*(1569.0*cos(q2 + q3) + 1700.0*cos(q2)))/4000+0.06*sin(q1)
    T[5][1] = -(sin(q1)*(1569.0*cos(q2 + q3) + 1700.0*cos(q2)))/4000-0.06*cos(q1)
    T[5][2] = 0.0894 - 0.425*sin(q2) - 0.39225*sin(q2 + q3)+z_sh
    T[6][0] = 0.10915*sin(q1) - 0.425*cos(q1)*cos(q2) + 0.0823*cos(q5)*sin(q1) + 0.39225*cos(q1)*sin(q2)*sin(q3) - 0.0823*cos(q2 + q3 + q4)*cos(q1)*sin(q5) + 0.09465*cos(q2 + q3)*cos(q1)*sin(q4) + 0.09465*sin(q2 + q3)*cos(q1)*cos(q4) - 0.39225*cos(q1)*cos(q2)*cos(q3)-0.05*sin(q1)
    T[6][1] = 0.39225*sin(q1)*sin(q2)*sin(q3) - 0.0823*cos(q1)*cos(q5) - 0.425*cos(q2)*sin(q1) - 0.10915*cos(q1) - 0.0823*cos(q2 + q3 + q4)*sin(q1)*sin(q5) + 0.09465*cos(q2 + q3)*sin(q1)*sin(q4) + 0.09465*sin(q2 + q3)*cos(q4)*sin(q1) - 0.39225*cos(q2)*cos(q3)*sin(q1)+0.05*cos(q1)
    T[6][2] = 0.09465*sin(q2 + q3)*sin(q4) - 0.425*sin(q2) - 0.39225*sin(q2 + q3) - sin(q5)*(0.0823*cos(q2 + q3)*sin(q4) + 0.0823*sin(q2 + q3)*cos(q4)) - 0.09465*cos(q2 + q3)*cos(q4) + 0.08945+z_sh
    
    return T

def get_velocity(q1,  q2,  q3,  q4,  q5,  q6, u_1,  u_2,  u_3,  u_4,  u_5,  u_6):
    U = [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    U[0][0] = 0.06*u_1*cos(q1)
    U[0][1] = 0.06*u_1*sin(q1)
    U[0][2] = 0
    U[1][0] = u_1*(0.14*cos(q1) + 0.2125*cos(q2)*sin(q1)) + 0.2125*u_2*cos(q1)*sin(q2)
    U[1][1] = u_1*(0.14*sin(q1) - 0.2125*cos(q1)*cos(q2)) + 0.2125*u_2*sin(q1)*sin(q2)
    U[1][2] = -0.2125*u_2*cos(q2)
    U[2][0] = u_1*(0.11*cos(q1) + 0.425*cos(q2)*sin(q1)) + 0.425*u_2*cos(q1)*sin(q2)
    U[2][1] = u_1*(0.11*sin(q1) - 0.425*cos(q1)*cos(q2)) + 0.425*u_2*sin(q1)*sin(q2)
    U[2][2] = -0.425*u_2*cos(q2)
    U[3][0] = 0.02*u_1*cos(q1) + 0.13075*u_1*cos(q2 + q3)*sin(q1) + 0.13075*u_2*sin(q2 + q3)*cos(q1) + 0.13075*u_3*sin(q2 + q3)*cos(q1) + 0.425*u_1*cos(q2)*sin(q1) + 0.425*u_2*cos(q1)*sin(q2)
    U[3][1] = 0.02*u_1*sin(q1) - 0.13075*u_1*cos(q2 + q3)*cos(q1) + 0.13075*u_2*sin(q2 + q3)*sin(q1) + 0.13075*u_3*sin(q2 + q3)*sin(q1) - 0.425*u_1*cos(q1)*cos(q2) + 0.425*u_2*sin(q1)*sin(q2)
    U[3][2] = - 1.0*u_2*(0.13075*cos(q2 + q3) + 0.425*cos(q2)) - 0.13075*u_3*cos(q2 + q3)
    U[4][0] = 0.02*u_1*cos(q1) + 0.2615*u_1*cos(q2 + q3)*sin(q1) + 0.2615*u_2*sin(q2 + q3)*cos(q1) + 0.2615*u_3*sin(q2 + q3)*cos(q1) + 0.425*u_1*cos(q2)*sin(q1) + 0.425*u_2*cos(q1)*sin(q2)
    U[4][1] = 0.02*u_1*sin(q1) - 0.2615*u_1*cos(q2 + q3)*cos(q1) + 0.2615*u_2*sin(q2 + q3)*sin(q1) + 0.2615*u_3*sin(q2 + q3)*sin(q1) - 0.425*u_1*cos(q1)*cos(q2) + 0.425*u_2*sin(q1)*sin(q2)
    U[4][2] = - 1.0*u_2*(0.2615*cos(q2 + q3) + 0.425*cos(q2)) - 0.2615*u_3*cos(q2 + q3)
    U[5][0] = u_1*(0.06*cos(q1) + 0.00025*sin(q1)*(1569.0*cos(q2 + q3) + 1700.0*cos(q2))) + 0.00025*u_2*cos(q1)*(1569.0*sin(q2 + q3) + 1700.0*sin(q2)) + 0.39225*u_3*sin(q2 + q3)*cos(q1)
    U[5][1] = u_1*(0.06*sin(q1) - 0.00025*cos(q1)*(1569.0*cos(q2 + q3) + 1700.0*cos(q2))) + 0.00025*u_2*sin(q1)*(1569.0*sin(q2 + q3) + 1700.0*sin(q2)) + 0.39225*u_3*sin(q2 + q3)*sin(q1)
    U[5][2] = - 1.0*u_2*(0.39225*cos(q2 + q3) + 0.425*cos(q2)) - 0.39225*u_3*cos(q2 + q3)
    U[6][0] = 0.05915*u_1*cos(q1) + 0.0823*u_1*cos(q1)*cos(q5) + 0.425*u_1*cos(q2)*sin(q1) + 0.425*u_2*cos(q1)*sin(q2) - 0.0823*u_5*sin(q1)*sin(q5) + 0.09465*u_2*cos(q2 + q3)*cos(q1)*cos(q4) + 0.09465*u_3*cos(q2 + q3)*cos(q1)*cos(q4) + 0.09465*u_4*cos(q2 + q3)*cos(q1)*cos(q4) - 0.09465*u_1*cos(q2 + q3)*sin(q1)*sin(q4) - 0.09465*u_1*sin(q2 + q3)*cos(q4)*sin(q1) - 0.09465*u_2*sin(q2 + q3)*cos(q1)*sin(q4) - 0.09465*u_3*sin(q2 + q3)*cos(q1)*sin(q4) - 0.09465*u_4*sin(q2 + q3)*cos(q1)*sin(q4) + 0.39225*u_1*cos(q2)*cos(q3)*sin(q1) + 0.39225*u_2*cos(q1)*cos(q2)*sin(q3) + 0.39225*u_2*cos(q1)*cos(q3)*sin(q2) + 0.39225*u_3*cos(q1)*cos(q2)*sin(q3) + 0.39225*u_3*cos(q1)*cos(q3)*sin(q2) - 0.39225*u_1*sin(q1)*sin(q2)*sin(q3) - 0.0823*u_5*cos(q2 + q3 + q4)*cos(q1)*cos(q5) + 0.0823*u_1*cos(q2 + q3 + q4)*sin(q1)*sin(q5) + 0.0823*u_2*sin(q2 + q3 + q4)*cos(q1)*sin(q5) + 0.0823*u_3*sin(q2 + q3 + q4)*cos(q1)*sin(q5) + 0.0823*u_4*sin(q2 + q3 + q4)*cos(q1)*sin(q5)
    U[6][1] = 0.05915*u_1*sin(q1) - 0.425*u_1*cos(q1)*cos(q2) + 0.0823*u_1*cos(q5)*sin(q1) + 0.0823*u_5*cos(q1)*sin(q5) + 0.425*u_2*sin(q1)*sin(q2) + 0.09465*u_1*cos(q2 + q3)*cos(q1)*sin(q4) + 0.09465*u_1*sin(q2 + q3)*cos(q1)*cos(q4) + 0.09465*u_2*cos(q2 + q3)*cos(q4)*sin(q1) + 0.09465*u_3*cos(q2 + q3)*cos(q4)*sin(q1) + 0.09465*u_4*cos(q2 + q3)*cos(q4)*sin(q1) - 0.39225*u_1*cos(q1)*cos(q2)*cos(q3) - 0.09465*u_2*sin(q2 + q3)*sin(q1)*sin(q4) - 0.09465*u_3*sin(q2 + q3)*sin(q1)*sin(q4) - 0.09465*u_4*sin(q2 + q3)*sin(q1)*sin(q4) + 0.39225*u_1*cos(q1)*sin(q2)*sin(q3) + 0.39225*u_2*cos(q2)*sin(q1)*sin(q3) + 0.39225*u_2*cos(q3)*sin(q1)*sin(q2) + 0.39225*u_3*cos(q2)*sin(q1)*sin(q3) + 0.39225*u_3*cos(q3)*sin(q1)*sin(q2) - 0.0823*u_1*cos(q2 + q3 + q4)*cos(q1)*sin(q5) - 0.0823*u_5*cos(q2 + q3 + q4)*cos(q5)*sin(q1) + 0.0823*u_2*sin(q2 + q3 + q4)*sin(q1)*sin(q5) + 0.0823*u_3*sin(q2 + q3 + q4)*sin(q1)*sin(q5) + 0.0823*u_4*sin(q2 + q3 + q4)*sin(q1)*sin(q5)
    U[6][2] = u_4*(0.09465*sin(q2 + q3 + q4) - 0.0823*cos(q2 + q3 + q4)*sin(q5)) - 1.0*u_3*(0.39225*cos(q2 + q3) - 0.09465*sin(q2 + q3 + q4) + 0.0823*cos(q2 + q3 + q4)*sin(q5)) - 1.0*u_2*(0.39225*cos(q2 + q3) + 0.425*cos(q2) - 0.09465*cos(q2 + q3)*sin(q4) - 0.09465*sin(q2 + q3)*cos(q4) + 0.0823*cos(q2 + q3)*cos(q4)*sin(q5) - 0.0823*sin(q2 + q3)*sin(q4)*sin(q5)) - 0.0823*u_5*sin(q2 + q3 + q4)*cos(q5)
    L = [0,0,0,0,0,0,0]

    for i in range(7):
        temp = U[i][0]*U[i][0]+U[i][1]*U[i][1]+U[i][2]*U[i][2]
        L[i] = temp
    return L

Rt = [0.15, 0.15, 0.15, 0.08, 0.08, 0.12, 0.1] #robot spheres
sph = [0.5510,0.6010,0.5010,0.5010,0.5010,0.5010,0.5010,0.5010,0.4510,0.4510,0.4810,0.4810,0.5510,0.6010] #human spheres
alpha = [2.79, 1.95, 1.00, 0.80, 0.65, 0.45, 0.35]

x = SX.sym('x',6) #joint poses
w = SX.sym('w',6) #joint vels
h = SX.sym('h',42) #human poses
n = SX.sym('n',6) #nn outputs
x = [0.0, -2.3, -1.1, -1.2, -1.2, 0.5]
h = [10.0517, 0.5220, 1.0895,10.0658,   0.4526,   0.8624, 10.0844,   0.7044,   0.9207, 10.2083,   0.3075,   1.0208, 10.0556,   0.6289,   0.7595, 10.2024,   0.2732,   0.8478, 10.0267,   0.5535,   0.5983, 10.1965,   0.2389,   0.6749, -10.0208,   0.3964,   0.5857, 10.0546,   0.2951,   0.6132, -10.1062,   0.2444,   0.5897, -10.0998,   0.3062,   0.5387, 10.1908,   0.5290,   1.0016,10.2106,   0.4602,   0.6915]
n = [1.0,-1.0,0.1,-0.1,0.1,1.0]

def cascadi_solv(n,data):
    x = data[0:6]
    h = data[6:48]
    f = (w[0]-n[0])**2+(w[1]-n[1])**2+(w[2]-n[2])**2+(w[3]-n[3])**2+(w[4]-n[4])**2+(w[5]-n[5])**2 #cost function
    T = get_cpose(x[0],x[1],x[2],x[3],x[4],x[5])
    L = get_velocity(x[0],x[1],x[2],x[3],x[4],x[5],w[0],w[1],w[2],w[3],w[4],w[5])
    H = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
         [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i in range(14):
        H[i][0] = h[0+i*3]
        H[i][1] = h[1+i*3]
        H[i][2] = h[2+i*3]
        H[i][3] = sph[i]

    J = 0
    g = []
    lbg = []
    ubg = []
    lbw = []
    ubw = []
    w0 = []

    ubw += [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lbw += [-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]
    w0 = n

    ctp = []
    for k in range(7):
        ctp.append(T[k][0])
        ctp.append(T[k][1])
        ctp.append(T[k][2])
        for j in range(14):
            b2 = (T[k][0] - H[j][0])*(T[k][0] - H[j][0])+(T[k][1] - H[j][1])*(T[k][1] - H[j][1])+(T[k][2] - H[j][2])*(T[k][2] - H[j][2])-(Rt[k] + H[j][3])*(Rt[k] + H[j][3])
            temp = [L[k] - alpha[k]*alpha[k]*b2]
            g += temp
            ubg += [0.0]
            lbg += [-10e6]

    # Create an NLP solver
    nlp = {'f': f, 'x': w, 'g': vertcat(*g)}
    opts = {}
    opts['ipopt.print_level'] = 0
    MySolver = "ipopt"
    # Allocate a solver
    solver = nlpsol("solver", MySolver, nlp, opts)
    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    # w_opt = sol['x'].full().flatten()
    a = sol["x"]

    L_2 = get_velocity(x[0],x[1],x[2],x[3],x[4],x[5],a[0],a[1],a[2],a[3],a[4],a[5])
    L_1 = get_velocity(x[0],x[1],x[2],x[3],x[4],x[5],n[0],n[1],n[2],n[3],n[4],n[5])

    min_dist = [10000]*7
    spheres_dist = [0]*7
    lin_vell_limit_arr = [10]*7
    cas_vel = [0]*7
    nn_vell = [0]*7
    for k in range(7):
        cas_vel[k] = sqrt(L_2[k])
        nn_vell[k] = sqrt(L_1[k])

    for j in range(7):
        robot = np.array([T[j][0],T[j][1],T[j][2]])
        for k in range(14):
            human = np.array([H[k][0],H[k][1],H[k][2]])
            b = np.array((robot[0:3] - human[0:3]))
            dist = np.linalg.norm(b)-Rt[j]-sph[k]
            if dist<min_dist[j]:
                min_dist[j] = dist
                spheres_dist[j] = Rt[j]+sph[k]

    for j in range(7):    
        sqrt_temp = (min_dist[j]+spheres_dist[j])*(min_dist[j]+spheres_dist[j])-spheres_dist[j]*spheres_dist[j]
        if sqrt_temp<0:
            lin_vell_limit_arr[j] = 0.00000000000
        else:
            lin_vell_limit_arr[j] = alpha[j]*sqrt(sqrt_temp)

    return a, cas_vel, nn_vell, lin_vell_limit_arr, ctp, min_dist

