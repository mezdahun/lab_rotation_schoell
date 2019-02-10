 
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 17:03:58 2018

@author: PROG
"""

import numpy as np
from scipy.integrate import odeint
from scipy.integrate import ode
import networkx as nx
import matplotlib.pyplot as plt
import time
import numba
from numba import jit


def ring_indeces(c,R,N):
    '''
    Returns the indeces of the presynaptic nodes in a ring configurarion,
    as the R neighbours of node c (center), where the number of nodes is N
    in a circular topology. The center node is not connected to itself.
    '''
    mini = c-R
    maxi = c+R+1
    inds = [i for i in range(mini,maxi)]
    for i,ind in enumerate(inds):
        if ind<0:
            inds[i]=ind+N
        inds[i] = ind % N
    inds.remove(c)
    return inds

def ring_conn_matrix(R,N):
    '''creates the adjacency matrix for ring topology of N nodes with coupling
    range R'''
    C = np.zeros([N,N])
    for k in range(N):
        C[k,ring_indeces(k,R,N)] = 1
    return C
    
def x_to_uv(x,N):
    '''returns state variable matrices from ode/odeint output'''
    try:
        N = int(x.shape[1]/2)
    except:
        N = int(x.shape[0]/2)
    u_m = np.zeros([x.shape[0],N])
    v_m = np.zeros([x.shape[0],N])
    for k in range(2*N):
        if k<N:
            u_m[:,k] = x[:,k]
        else:
            v_m[:,k-N] = x[:,k]
    return [u_m, v_m]

def return_count(u_m_ode, u_thr=-1.5): 
    count = np.zeros(N)
    state_prev_u = u_m_ode[0,:]
    for i in range(1,u_m_ode.shape[0]):
        state_now_u = u_m_ode[i,:]
        cond_prev = np.where(state_prev_u<=u_thr)
        cond_now = np.where(state_now_u>u_thr)
        cond = np.intersect1d(cond_prev, cond_now)
        count[cond]+=1
        state_prev_u = state_now_u.copy()
    return count

def FHN_ode_fast(t,x,eps,a,N,sigma_intra,R,phi,C_intra,sig_ij,x_del,sig_ij2=0, x_del2=0, flag2 = -1):
    if flag2==-1:
        sig_ij2=0
        x_del2=np.empty_like(x_del)
    return FHN_ode_fast_helper(t,x,eps,a,N,sigma_intra,R,phi,C_intra, sig_ij,x_del,flag2, sig_ij2,x_del2)

@jit(nopython=True, cache=True)
def FHN_ode_fast_helper(t,x,eps,a,N,sigma_intra,R,phi,C_intra,sig_ij,x_del, flag2, sig_ij2,x_del2):
    '''
    Implements a set of Fitz-Hugh_nagumo dynamical nodes with ring coupling
    for ode (and not odeint) method. The only difference wrt FHN is the order
    of t and x in the signature
    
    parameters:
        x = [u, v]      : dynamical parameter state vector
        t               : time
        eps             : time-scale separator of u and v (Epsilon)
        a               : threshold parameter
        N               : number of nodes
        sigma_intra     : coupling strength (intralayer)
        R               : coupling range
        phi             : coupling phase
        C_intra         : intralayer adjacency matrix
        sig_ij          : interlayer coupling strength
        x_del           : delayed interlayer state, same structure as x
    '''
    # w is scaling parameter, H is diffuse coupling matrix
    w = sigma_intra/(2*R)
    r1 = np.array([(1./eps)*np.cos(phi), (1./eps)*np.sin(phi)])
    r2 = np.array([-np.sin(phi),         np.cos(phi)])
    H = np.vstack((r1,r2))
   # H = np.array([[(1./eps)*np.cos(phi), (1./eps)*np.sin(phi)],
   #               [-np.sin(phi),         np.cos(phi)]])    
    # the local dynamics is second order (excitation, inhibition)
    u = x.reshape(2,N)[0]
    v = x.reshape(2,N)[1]
    # getting delayed values
    #[u_del, v_del] = x_to_uv(x_del, N)            
    u_del = x_del.reshape(2,N)[0]
    v_del = x_del.reshape(2,N)[1]
    if flag2>0:
        u_del2 = x_del2.reshape(2,N)[0]
        v_del2 = x_del2.reshape(2,N)[1]
    # we have N nodes need to create a dudt and dvdt for each
    dudt = np.empty(N)
    dvdt = np.empty(N) 
    transl = [n for n in range(N)]
    for k in range(N):
        # first we have to calculate the INTRAlayer contribution
        # here the coupling is ring like with R coupling range
        subtraction_fac = 2*R*np.dot(H,np.array([u[k],v[k]]))
        cont_inter = np.zeros(2)
        diff_inter = np.array([u_del[k],v_del[k]])
        cont_inter = np.dot(H,diff_inter)-subtraction_fac/(2*R)
        if flag2==np.int64(1):
            cont_inter2 = np.zeros(2)
            diff_inter2 = np.array([u_del2[k],v_del2[k]])
            cont_inter2 = np.dot(H,diff_inter2)-subtraction_fac/(2*R)
        if k == 0:
            diff = np.zeros_like(np.array([u[0],v[0]]))
            indeces_connected_intra = np.where(C_intra[k,:] != 0)[0]
            cont_intra = np.zeros(2)
            for l in indeces_connected_intra:
                diff += np.array([u[l],v[l]])
        else:
            plus_diff_indeces = [k-1, (k+R)%N]
            minus_diff_indeces = [k, transl[k-R-1]]
            for ind in plus_diff_indeces:            
                diff += np.array([u[ind],v[ind]])
            for ind in minus_diff_indeces:
                diff -= np.array([u[ind],v[ind]])
        # then we can add it to the local dynamics
        cont_intra = np.dot(H,diff)-subtraction_fac#/(2*R)
        if flag2==np.int64(1):
            dudt[k] = (1./eps)*(u[k]-((u[k]**3)/3)-v[k]) + w*cont_intra[0] + 0.5*sig_ij*cont_inter[0] + 0.5*sig_ij2*cont_inter2[0]
            dvdt[k] = u[k] + a + w*cont_intra[1] + 0.5*sig_ij*cont_inter[1] + 0.5*sig_ij2*cont_inter2[1]
        else:
            dudt[k] = (1./eps)*(u[k]-((u[k]**3)/3)-v[k]) + w*cont_intra[0] + sig_ij*cont_inter[0]
            dvdt[k] = u[k] + a + w*cont_intra[1] + sig_ij*cont_inter[1]
    return np.hstack((dudt,dvdt))

def tau_i(tau, dt):
    '''calculates index differenec from time difference with given dt'''
    return int(np.round(tau/dt))

def get_initial_conditions(N):
    '''initial condition randomly distributed on the r=2 circle in state space'''
    r = 2
    phi_rand    = np.random.uniform(0,2*np.pi,N)
    u_0         = r*np.cos(phi_rand)
    v_0         = r*np.sin(phi_rand)
    x_0         = np.concatenate([u_0, v_0])
    return x_0

'''@jit(nopython=True, cache=True)
def split_list(n):
    """will return the list index"""
    return [(x+1) for x,y in zip(n, n[1:]) if y-x != 1]

@jit(nopython=True, cache=True)
def get_sub_list(my_list):
    """will split the list base on the index"""
    my_index = split_list(my_list)
    output = []
    prev = 0
    for index in my_index:
        new_list = [ x for x in my_list[prev:] if x < index]
        output.append(new_list)
        prev += len(new_list)
    output.append([ x for x in my_list[prev:]])
    return output

def consecutive(data, stepsize=1):
    #retruns the following consecutive indeces from detrender:
    #source: https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)'''


@jit(nopython=True, cache=True)
def split_list(n):
    """will return the list index"""
    return [(x+1) for x,y in zip(n, n[1:]) if y-x != 1]

@jit(nopython=True, cache=True)
def get_sub_list(my_list):
    """will split the list base on the index"""
    my_index = split_list(my_list)
    output = []
    prev = 0
    for index in my_index:
        new_list = np.array([ x for x in my_list[prev:] if x < index])
        output.append(new_list)
        prev += len(new_list)
    output.append(np.array([ x for x in my_list[prev:]]))
    return output

def consecutive(data, stepsize=1):
    '''retruns the following consecutive indeces from detrender:
    source: https://stackoverflow.com/questions/7352684/how-to-find-the-groups-of-consecutive-elements-from-an-array-in-numpy'''
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def get_groups(x, N):
    norm = np.linalg.norm(x.reshape(2,N),axis=0)
    b = np.zeros(len(norm))
    b[0:-1] = -np.diff(norm)
    b[-1] = norm[0]-norm[len(norm)-1]
    detrender = b#.abs(b)
    groups = get_sub_list(np.where(detrender<0.25)[0])
    return groups

def detrend(x,N):
    '''detrending x as in the paper SWA18'''
    u = x.reshape(2,N)[0]
    v = x.reshape(2,N)[1]
    groups = get_groups(x, N)
    catchflag = False
    for group in groups:
        if group[-1]==N-1:
            u = np.roll(u,len(group))
            v = np.roll(v,len(group))
            catchflag = True
            break
    if catchflag:            
        groups = get_groups(np.hstack((u,v)), N)
    largest_group = groups[np.argmax([len(group) for group in groups])]
    c = largest_group[int(len(largest_group)/2)]
    u_d = np.roll(u,-c)
    v_d = np.roll(v,-c)
    return np.hstack((u_d,v_d))

def detrend_old(x):
    '''detrending x as in the paper SWA18'''
    u = x.reshape(2,N)[0]
    #plt.plot(u)
    v = x.reshape(2,N)[1]
    b = np.zeros(len(u))
    b[0:-1] = -np.diff(u)
    b[-1] = u[0]-u[len(u)-1]
    detrender = np.abs(b)
    groups = get_sub_list(np.where(detrender<0.25)[0])
    largest_group = groups[np.argmax([len(group) for group in groups])]
    c = largest_group[int(len(largest_group)/2)]
    groups.remove(largest_group)
    if len(groups)!=0:
        second_largest = groups[np.argmax([len(group) for group in groups])]
        if second_largest[-1]==N-1:
            uniroll = second_largest[0]
        else:
            uniroll = 0
    else:
        uniroll = 0
    #plt.scatter(c,3)
    u_d = np.roll(u,-(c-int(uniroll/2)))
    v_d = np.roll(v,-(c-int(uniroll/2)))
    return np.hstack((u_d,v_d))

def E_loc(xj,xi,t_max):
    '''returns the local synchronisation error for all node k between layer j and layer i.
    here xj is list xj = [uj, vj]'''
    diff = []
    diff.append(xj[0]-xi[0])
    diff.append(xj[1]-xi[1])
    norm = np.linalg.norm(diff,axis=0)
    #return (1./t_max)*np.sum(norm, axis = 0)
    return np.mean(norm, axis=0)

def E_glob(uj,ui,t_max):
    '''Returns the global synchronisation error between layer j and layer i'''
    return (1./ui.shape[1])*np.sum(E_loc(uj,ui,t_max))

def E_loc_norm(xj,xi,t_max):
    '''returns the local synchronisation error for all node k between layer j and layer i.
    here xj is list xj = [uj, vj]'''
    diff = []
    diff.append(xj[0]-xi[0])
    diff.append(xj[1]-xi[1])
    norm = np.linalg.norm(diff,axis=0)
    return norm

def omega(u, t_max):
    count= return_count(u)
    return 2*np.pi*count/t_max

@jit(nopython=True, cache=True)
def selfrollone(x):
    x_p = np.zeros_like(x)
    x_p[0:-1] = x[1::]
    x_p[-1] = x[0]
    return x_p

def snapshot(u1,u2,u3,i,t, file_to):
    fig, ax = plt.subplots(3, 1, figsize = [10,8], sharex = True, sharey = True)
    plt.xlim([0,N])
    plt.axes(ax[0])
    plt.plot(u3[i],'b.')
    plt.title('Snapshot of state variable $u_k^i(t)$ at $t_s$={:.2f}, $i=1,2,3$'.format(t[i]))
    plt.ylabel('$u_k^3(t_s)$')
    ax2 = ax[0].twinx()
    plt.axes(ax2)
    plt.yticks([])
    ax2.set_ylabel('Layer 3', color='blue', rotation=270, va='bottom')

    plt.axes(ax[1])
    plt.plot(u2[i],'r.', label='Laxer 2')
    plt.ylabel('$u_k^2(t_s)$')
    ax2 = ax[1].twinx()
    plt.axes(ax2)
    plt.yticks([])
    ax2.set_ylabel('Layer 2', color='red', rotation=270, va='bottom')

    plt.axes(ax[2])
    plt.plot(u1[i],'b.',label="Layer 1")
    plt.ylabel('$u_k^1(t_s)$')
    ax2 = ax[2].twinx()
    plt.axes(ax2)
    plt.yticks([])
    ax2.set_ylabel('Layer 1', color='blue', rotation=270, va='bottom')
    plt.axes(ax[2])
    plt.xlabel('Node # $k$')
    plt.savefig(file_to)
    #plt.show()

def omega_E(omega1,omega2,omega3,E12,E13, file_to):  
    fig, ax = plt.subplots(3, 1, figsize = [10,8], sharex = True, sharey = True)
    plt.xlim([0,N])
    plt.axes(ax[0])
    plt.title('Mean phase velocity $\omega^i(k)$ and Local synchronization error $E^{ij}_k$, $i,j\in\{1,2,3\}$')
    plt.ylabel('$\omega^3(k)$', color='blue')
    plt.plot(omega3, 'blue', label='$\omega^3(k)$')
    #plt.ylim([2,8])

    plt.axes(ax[1])
    plt.plot(omega2, 'blue', label='$\omega^2(k)$')
    plt.ylabel('$\omega^2(k)$', color='blue')
    ax2 = ax[1].twinx()
    plt.axes(ax2)
    plt.ylim([0,3])
    plt.yticks([0,3])
    plt.plot(E12,'orange', label='$E^{12}_k$')
    #plt.plot(E_loc([u2,v2],[u1,v1], t_max),'orange', label='$E^{12}_k$')
    ax2.set_ylabel('$E^{12}_k$', color='orange', rotation=270, va='bottom')

    plt.axes(ax[2])
    plt.plot(omega1, 'blue', label='$\omega^1(k)$')
    plt.ylabel('$\omega^1(k)$', color='blue')
    ax2 = ax[2].twinx()
    plt.axes(ax2)
    plt.ylim([0,3])
    plt.yticks([0,3])
    plt.plot(E13,'orange', label='$E^{12}_k$')
    #plt.plot(E_loc([u3,v3],[u1,v1], t_max),'orange', label='$E^{13}_k$')
    ax2.set_ylabel('$E^{13}_k$', color='orange', rotation=270, va='bottom')
    plt.axes(ax[2])
    plt.xlabel('Node # $k$')
    plt.savefig(file_to)
    #plt.show()


#####################PARAMETERS################################
#figure name in article
fign = "2c"
# Local and global dynamics
a1, a2, a3  = 0.5, 0.5, 0.5    # Threshold parameters
epsilon     = 0.05             # Time-scale separation of u and v
N           = 500              # Num of nodes in a layer
phi         = np.pi/2 - 0.1    # Coupling phase
R1, R2, R3  = 170, 170, 170    # Coupling range
S1, S2, S3  = 0.2, 0.2, 0.2    # Coupling strength (intralayer)

# interlayer coupling strength
sigma_12    = 0.025            # (1->2) Interlayer coupling,
sigma_21    = 0.025           # (2->1) Interlayer coupling,

sigma_13    = 0                # (1->3) Interlayer coupling,
sigma_31    = 0                # (3->1) Interlayer coupling,

sigma_23    = 0.025           # (2->3) Interlayer coupling
sigma_32    = 0.025           # (3->2) Interlayer coupling

# intralayer coupling matrices
C1, C2, C3   = ring_conn_matrix(R1,N), ring_conn_matrix(R2,N), ring_conn_matrix(R3,N)

# integration parameters
t_max       = 600
t_res       = 20
t_n         = t_max*t_res
t           = np.linspace(0,t_max,t_n)
dt          = t[1] - t[0]
int_type    = "dop853"
t_calc = 50

# interlayer delay
tau_12      = 0.4             # (1->2) Interlayer delay,
tau_21      = 0.4              # (2->1) Interlayer delay,

tau_23      = 0.4              # (2->3) Interlayer delay,
tau_32      = 0.4              # (3->2) Interlayer delay,

tau_12, tau_21, tau_23, tau_32 = tau_i(tau_12,dt), tau_i(tau_21,dt), tau_i(tau_23,dt), tau_i(tau_32,dt)
#tau_m = np.max([tau_12, tau_21, tau_23, tau_32])
tau_calc = tau_i(t_calc,dt)
tau_m = 200 #last tau_m number of timesteps which is going to be saved

# initial conditions
L1_0 = get_initial_conditions(N)
L2_0 = get_initial_conditions(N) 
L3_0 = get_initial_conditions(N)
#L1_0 = np.load('x1_{}.npy'.format(fign))[-1,]
#L2_0 = np.load('x2_{}.npy'.format(fign))[-1,]
#L3_0 = np.load('x3_{}.npy'.format(fign))[-1,]
detrend_s = False

##############################################################

# using ode instead of ode_int to access the state evolution history
x_ode_1       = np.zeros([tau_m+1,2*N])
x_ode_2       = np.zeros([tau_m+1,2*N])
x_ode_3       = np.zeros([tau_m+1,2*N])
#for detrending
x_ode_1_d       = np.zeros([tau_m+1,2*N])
x_ode_2_d       = np.zeros([tau_m+1,2*N])
x_ode_3_d       = np.zeros([tau_m+1,2*N])
# initializing omegas
count1 = np.zeros(N)
count2 = np.zeros(N)
count3 = np.zeros(N)
# initializing errors
E12 = np.zeros(N)
E13 = np.zeros(N)
# setting up the solver
L1 = ode(FHN_ode_fast).set_integrator(int_type)
L1.set_initial_value(L1_0)
L2 = ode(FHN_ode_fast).set_integrator(int_type)
L2.set_initial_value(L2_0)
L3 = ode(FHN_ode_fast).set_integrator(int_type)
L3.set_initial_value(L3_0)





ode_start = time.time()
for t_i in range(len(t)):
    if int(float(t_i)/float(len(t))*100)%5==0:
      print('{:.2f}% done'.format(float(t_i)/float(len(t))*100))
    L1.set_f_params(epsilon,a1,N,S1,R1,phi,C1,sigma_21, x_ode_2[-tau_21-1,:], 0, 0, -1)#,False,None,None)
    L1.integrate(L1.t+dt)
    x_ode_1[-1,:] = L1.y
    x_ode_1_d[-1,:] = detrend(L1.y, N)
    
    L2.set_f_params(epsilon,a2,N,S2,R2,phi,C2,sigma_12,x_ode_1[-tau_12-1,:],sigma_32,x_ode_3[-tau_32-1,:],1)
    L2.integrate(L2.t+dt)
    x_ode_2[-1,:] = L2.y
    x_ode_2_d[-1,:] = detrend(L2.y, N)
    
    L3.set_f_params(epsilon,a3,N,S3,R3,phi,C3,sigma_23, x_ode_2[-tau_23-1,:],0,0,-1)
    L3.integrate(L3.t+dt)
    x_ode_3[-1,:] = L3.y
    x_ode_3_d[-1,:] = detrend(L3.y, N)
    
    #adding omegas
    [u1, v1] = x_to_uv(x_ode_1_d,N)
    [u2, v2] = x_to_uv(x_ode_2_d,N)
    [u3, v3] = x_to_uv(x_ode_3_d,N)
    if t_i>tau_calc:#2:
        count1 += return_count(u1[-2::])
        count2 += return_count(u2[-2::])
        count3 += return_count(u3[-2::])
        E12+=E_loc_norm([u1[-1],v1[-1]],[u2[-1],v2[-1]], t_max)
        E13+=E_loc_norm([u1[-1],v1[-1]],[u3[-1],v3[-1]], t_max)
        
    x_ode_1 = selfrollone(x_ode_1)
    x_ode_2 = selfrollone(x_ode_2)
    x_ode_3 = selfrollone(x_ode_3)
    x_ode_1_d = selfrollone(x_ode_1_d)
    x_ode_2_d = selfrollone(x_ode_2_d)
    x_ode_3_d = selfrollone(x_ode_3_d)
    
omega1 = 2*np.pi*count1/(t_max-t_calc)
omega2 = 2*np.pi*count2/(t_max-t_calc)
omega3 = 2*np.pi*count3/(t_max-t_calc)
E12/=(len(t)-tau_calc)
E13/=(len(t)-tau_calc)
ode_end = time.time()
ode_perf = ode_end - ode_start
print("ODE elapsed time: {}s, #t={}, N={}".format(ode_perf, t_max, N))

np.save('L1_{}.npy'.format(fign), L1_0)
np.save('L2_{}.npy'.format(fign), L2_0)
np.save('L3_{}.npy'.format(fign),L3_0)
np.save('E12_{}.npy'.format(fign),E12)
np.save('E13_{}.npy'.format(fign),E13)
np.save('omega1_{}.npy'.format(fign),omega1)
np.save('omega2_{}.npy'.format(fign),omega2)
np.save('omega3_{}.npy'.format(fign),omega3)

[u1, v1] = x_to_uv(x_ode_1,N)
x1 = np.hstack((u1,v1))
[u2, v2] = x_to_uv(x_ode_2,N)
x2 = np.hstack((u2,v2))
[u3, v3] = x_to_uv(x_ode_3,N)
x3 = np.hstack((u3,v3))

u = np.hstack((u1,u2,u3))

np.save('u_all_{}.npy'.format(fign),u)
np.save('x1_{}.npy'.format(fign),x1)
np.save('x2_{}.npy'.format(fign),x2)
np.save('x3_{}.npy'.format(fign),x3)

snapshot(u1,u2,u3,-1,t, 'Snapshot_{}.png'.format(fign))
omega_E(omega1,omega2,omega3,E12,E13, 'omega_E_{}.png'.format(fign))
