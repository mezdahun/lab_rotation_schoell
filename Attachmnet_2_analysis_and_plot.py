#!/usr/bin/env python
# coding: utf-8



import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import time
import pandas as pd
import os


def cantor_ring(b_0, n, break_symmetry = True):
    '''b_o is an initiation string,
    the function generates a circular cantor fractal topology of depth n with initiation string b_0
    Option: symmetry breaking: adding zero to self connections'''
    b = len(b_0)
    repzero = b*'0'
    repone = b_0
    for i in range(n):
        b_0 = b_0.replace('0',repzero)
        b_0 = b_0.replace('1',repone)
    if break_symmetry:
        b_0 = ''.join(['0',b_0])
    return b_0

def cantor_matrix(b_0, n, break_symmetry = True):
    row = cantor_ring(b_0, n, break_symmetry = True)
    row = np.array([int(i) for i in list(row)])
    N = len(row)
    G = np.zeros([N,N])
    for k in range(N):
        G[k] = row
        row = np.roll(row,1)
    return G

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
    N = u_m_ode.shape[1]
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

#@jit(nopython=True, cache=True)
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
    H = np.array([[(1./eps)*np.cos(phi), (1./eps)*np.sin(phi)],
                  [-np.sin(phi),         np.cos(phi)]])    
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
            cont_inter2 = np.dot(H,diff_inter)-subtraction_fac/(2*R)
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
        cont_intra = np.dot(H,diff)-subtraction_fac
        if flag2==np.int64(1):
            dudt[k] = (1./eps)*(u[k]-((u[k]**3)/3)-v[k]) + w*cont_intra[0] + sig_ij*cont_inter[0] + sig_ij2*cont_inter2[0]
            dvdt[k] = u[k] + a + w*cont_intra[1] + sig_ij*cont_inter[1] + sig_ij2*cont_inter2[1]
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

#@jit(nopython=True, cache=True)
def split_list(n):
    """will return the list index"""
    return [(x+1) for x,y in zip(n, n[1:]) if y-x != 1]

#@jit(nopython=True, cache=True)
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

def E_loc_norm(xj,xi,t_max):
    '''returns the local synchronisation error for all node k between layer j and layer i.
    here xj is list xj = [uj, vj]'''
    diff = []
    diff.append(xj[0]-xi[0])
    diff.append(xj[1]-xi[1])
    norm = np.linalg.norm(diff,axis=0)
    return norm

def E_loc(xj,xi,t_max):
    '''returns the local synchronisation error for all node k between layer j and layer i.
    here xj is list xj = [uj, vj]'''
    diff = []
    diff.append(xj[0]-xi[0])
    diff.append(xj[1]-xi[1])
    norm = np.linalg.norm(diff,axis=0)
    #return (1./t_max)*np.sum(norm, axis = 0)
    return np.mean(norm, axis=0)
    
def E_glob(xj,xi,t_max):
    '''Returns the global synchronisation error between layer j and layer i'''
    return (1./ui.shape[1])*np.sum(E_loc(xj,xi,t_max))

#@jit(nopython=True, cache=True)
def selfrollone(x):
    x_p = np.zeros_like(x)
    x_p[0:-1] = x[1::]
    x_p[-1] = x[0]
    return x_p

def omega(u, t_max):
    count= return_count(u)
    return 2*np.pi*count/t_max

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
    N=500
    fig, ax = plt.subplots(3, 1, figsize = [7,4], sharex = True, sharey = True)
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
    plt.ylim([0.1,3])
    plt.yticks([0,3])
    plt.plot(E12,'orange', label='$E^{12}_k$')
    #plt.plot(E_loc([u2,v2],[u1,v1], t_max),'orange', label='$E^{12}_k$')
    ax2.set_ylabel('$E^{12}_k$', color='orange', rotation=270, va='bottom')

    plt.axes(ax[2])
    plt.plot(omega1, 'blue', label='$\omega^1(k)$')
    plt.ylabel('$\omega^1(k)$', color='blue')
    ax2 = ax[2].twinx()
    plt.axes(ax2)
    plt.ylim([0.1,3])
    plt.yticks([0,3])
    plt.plot(E13,'orange', label='$E^{12}_k$')
    #plt.plot(E_loc([u3,v3],[u1,v1], t_max),'orange', label='$E^{13}_k$')
    ax2.set_ylabel('$E^{13}_k$', color='orange', rotation=270, va='bottom')
    plt.axes(ax[2])
    plt.xlabel('Node # $k$')
    #plt.savefig(file_to)
    plt.show()

def read_in_single_data(d):
    '''function to read in sigle pieces of data from a folder as the output of:
    
    [OMEGA, E, u], where:
    OMEGA = np.vstack(omega1, omega2, omega2) #shape is 3xN
    E     = np,vstack(E12, E13) #shape is 2xN
    u = u_all as given by simulation #shape is as returned by simulation it is tau_m x (3xN)'''
    os.chdir(d)
    omega1 = np.load('omega1.npy')
    omega2 = np.load('omega2.npy')
    omega3 = np.load('omega3.npy')
    omega = np.vstack((omega1, omega2, omega3))
    e12 = np.load('E12.npy')
    e13 = np.load('E13.npy')
    E = np.vstack((e12, e13))
    print(np.shape(E))
    u = np.load('u_all.npy')
    print(np.shape(u))
    return [omega, E, u]

def read_in_single(d):
    '''function to read in sigle pieces of data from a folder as the output of:
    
    [OMEGA, E, u], where:
    OMEGA = np.vstack(omega1, omega2, omega2) #shape is 3xN
    E     = np,vstack(E12, E13) #shape is 2xN
    u = u_all as given by simulation #shape is as returned by simulation it is tau_m x (3xN)'''
    os.chdir(d)
    omega1 = np.load('omega1_{}.npy'.format(d))
    omega2 = np.load('omega2_{}.npy'.format(d))
    omega3 = np.load('omega3_{}.npy'.format(d))
    omega = np.vstack((omega1, omega2, omega3))
    e12 = np.load('E12_{}.npy'.format(d))
    e13 = np.load('E13_{}.npy'.format(d))
    E = np.vstack((e12, e13))
    print(np.shape(E))
    u = np.load('u_all_{}.npy'.format(d))
    print(np.shape(u))
    return [omega1,omega2, omega3, e12, e13, u]

def read_in_data(parent_dir):
    '''function to automatically read in the iterated data along changing R2 from the different folders'''
    
    #parent_dir = '/home/davidm/schoell_lab'
    os.chdir(parent_dir)
    directories = [os.path.join(parent_dir, o) for o in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir,o))]
    directories = [d for d in directories if d.find('R2_')!=-1]
    data_dict = {}
    for d in directories:
        os.chdir(d)
        # do something
        R2 = int(d.split('R2_')[-1])
        data_dict[R2] = read_in_single_data(d) # = [omega, E, u]
    os.chdir(parent_dir)
    return data_dict

def show_omega_all(data):
    '''data as returned by read_in_data'''
    fig, ax = plt.subplots(3, 1, figsize = [10,8], sharex = True, sharey = True)
    n = len(data)
    Rs = []
    for k, v in sorted(data.items()):
        Rs.append(k)
    Rs = Rs/np.max(Rs)
    colors = pl.cm.jet(Rs)#np.linspace(0,1,n))
    j = 0
    for k, v in sorted(data.items()):
        for i in range(3):
            plt.axes(ax[i])
            plt.plot(v[0][i,:], label=str(k), color=colors[j])
        j+=1
        
    plt.axes(ax[2])
    plt.xlabel('Node # $k$')
    plt.ylabel('$\omega^3(k)$', color='blue')
    
    plt.axes(ax[1])
    plt.ylabel('$\omega^2(k)$', color='blue')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='$R_2$ values')
    
    plt.axes(ax[0])
    plt.title('Mean phase velocity $\omega^i(k)$ for changing relay coupling range $R_2$, $i \in\{1,2,3\}$')
    plt.ylabel('$\omega^1(k)$', color='blue')
    
    plt.show()
    
def show_E_all(data):
    '''data as returned by read_in_data'''
    fig, ax = plt.subplots(1, 1, figsize = [10,8])
    plt.axes(ax)
    plt.xlabel('Node # $k$')
    plt.ylabel('$E_{ij}$', color='orange')
    plt.title('Local synchronisation errors $E_{12}$ (dashed) and $E_{13}$ (solid), for changing $R_2$')
    Rs = []
    for k, v in sorted(data.items()):
        Rs.append(k)
    Rs = Rs/np.max(Rs)
    colors_12 = pl.cm.spring(Rs)
    colors_13 = pl.cm.autumn(Rs)
    j = 0
    for k, v in sorted(data.items()):
        plt.plot(v[1][0,:], color=colors_13[j], ls='--')
        plt.plot(v[1][1,:], label=str(k), color=colors_13[j])
        j+=1
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='$R_2$ values')
    plt.show()
    
def mean_std_omega(data):
    '''showing mean and std of omegas'''        
    n = len(data)
    N = np.shape(data[1][0][0,:])[0]
    omegas = np.zeros([3,n,N])
    j = 0
    for k, v in sorted(data.items()):
        for i in range(3):
            omegas[i,j,:] = v[0][i,:]
        j+=1
    m_omegas = np.mean(omegas, axis=1)
    s_omegas = np.std(omegas, axis=1)
    x = range(1,501)
    
    fig, ax = plt.subplots(3, 1, figsize = [10,8], sharex = True, sharey = True)
    for i in range(3):
        plt.axes(ax[i])
        plt.fill_between(x, m_omegas[i]-s_omegas[i], m_omegas[i]+s_omegas[i], facecolor='#089FFF', label='std($\omega$)', alpha=0.2)
        #plt.errorbar(x, m_omegas[0], s_omegas[0], ecolor='gray')
        plt.plot(x,m_omegas[i], c='blue', label='mean $\omega$')
        
    plt.axes(ax[2])
    plt.xlabel('Node # $k$')
    plt.ylabel('$\omega^3(k)$', color='blue')
    
    plt.axes(ax[1])
    plt.ylabel('$\omega^2(k)$', color='blue')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='legend')
    
    plt.axes(ax[0])
    plt.title('Mean and STD of mean phase velocity $\omega^i(k)$ for changing $R_2$')
    plt.ylabel('$\omega^1(k)$', color='blue')
    plt.show()
    return omegas
    
def mean_std_E(data):
    n = len(data)
    N = np.shape(data[1][1][0,:])[0]
    Es = np.zeros([2,n,N])
    j = 0
    for k, v in sorted(data.items()):
        for i in range(2):
            Es[i,j,:] = v[1][i,:]
        j+=1
    
    m_Es = np.mean(Es, axis=1)
    s_Es = np.std(Es, axis=1)
    x = range(1,501)
    
    fig, ax = plt.subplots(1, 1, figsize = [10,8])
    plt.axes(ax)
    plt.xlabel('Node # $k$')
    plt.ylabel('$E_{ij}$', color='orange')
    plt.title('Local synchronisation errors $E_{12}$ (dashed) and $E_{13}$ (solid), for changing $R_2$')
    
    
    plt.fill_between(x, m_Es[0]-s_Es[0], m_Es[0]+s_Es[0], facecolor='orange', alpha=0.2, linestyle='--')
    plt.plot(x,m_Es[0], c='orange', ls='--')
    
    plt.fill_between(x, m_Es[1]-s_Es[1], m_Es[1]+s_Es[1], facecolor='orange', label='std($E_{ij}$)', alpha=0.2)
    plt.plot(x,m_Es[1], c='orange', label='mean $E_{ij}$')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='legend')
    
    #plt.show()
    return Es
    
def mean_std_omega_E(data):
    n = len(data)
    N = np.shape(data[1][1][0,:])[0]
    Es = np.zeros([2,n,N])
    j = 0
    for k, v in sorted(data.items()):
        for i in range(2):
            Es[i,j,:] = v[1][i,:]
        j+=1
    
    m_Es = np.mean(Es, axis=1)
    s_Es = np.std(Es, axis=1)
    
    n = len(data)
    N = np.shape(data[1][0][0,:])[0]
    omegas = np.zeros([3,n,N])
    j = 0
    for k, v in sorted(data.items()):
        for i in range(3):
            omegas[i,j,:] = v[0][i,:]
        j+=1
    m_omegas = np.mean(omegas, axis=1)
    s_omegas = np.std(omegas, axis=1)
    x = range(1,501)
    
    
    fig, ax = plt.subplots(3, 1, figsize = [10,8], sharex = True, sharey = True)
    plt.xlim([0,N])
    plt.axes(ax[0])
    plt.title('Mean and STD of mean phase velocity $\omega^i(k)$ and Local synch error $E_{ij}$ for changing $R_2$')
    plt.ylabel('$\omega^3(k)$', color='blue')
    plt.fill_between(x, m_omegas[2]-s_omegas[2], m_omegas[2]+s_omegas[2], facecolor='#089FFF', alpha=0.2, label='std($\omega$)')
    #plt.errorbar(x, m_omegas[0], s_omegas[0], ecolor='gray')
    plt.plot(x,m_omegas[2], c='blue', label='mean $\omega$')
    #plt.legend(loc='center left')
    
    
    plt.axes(ax[1])
    plt.ylabel('$\omega^2(k)$', color='blue')
    plt.fill_between(x, m_omegas[1]-s_omegas[1], m_omegas[1]+s_omegas[1], facecolor='#089FFF', alpha=0.2)
    #plt.errorbar(x, m_omegas[0], s_omegas[0], ecolor='gray')
    plt.plot(x,m_omegas[1], c='blue')
    ax2 = ax[1].twinx()
    plt.axes(ax2)
    plt.ylim([0,3])
    plt.yticks([0,3])
    plt.fill_between(x, m_Es[0]-s_Es[0], m_Es[0]+s_Es[0], facecolor='orange', alpha=0.2, linestyle='--', label='$STD(E_{12})$')
    plt.plot(x,m_Es[0], c='orange', ls='--', label='mean $E_{12}$')    
    ax2.set_ylabel('$E^{12}_k$', color='orange', rotation=270, va='bottom')    
    #plt.legend(loc='center left')

    plt.axes(ax[2])
    plt.ylabel('$\omega^1(k)$', color='blue')
    plt.fill_between(x, m_omegas[0]-s_omegas[0], m_omegas[0]+s_omegas[0], facecolor='#089FFF', label='std($\omega$)', alpha=0.2)
    #plt.errorbar(x, m_omegas[0], s_omegas[0], ecolor='gray')
    plt.plot(x,m_omegas[0], c='blue', label='mean $\omega$')
    ax2 = ax[2].twinx()
    plt.axes(ax2)
    plt.ylim([0,3])
    plt.yticks([0,3])
    plt.fill_between(x, m_Es[1]-s_Es[1], m_Es[1]+s_Es[1], facecolor='orange', label='std($E_{13}$)', alpha=0.2)
    plt.plot(x,m_Es[1], c='orange', label='mean $E_{13}$')
    #plt.plot(E_loc([u3,v3],[u1,v1], t_max),'orange', label='$E^{13}_k$')
    ax2.set_ylabel('$E^{13}_k$', color='orange', rotation=270, va='bottom')
    #plt.legend(loc='center left')
    plt.axes(ax[2])
    plt.xlabel('Node # $k$')  
    
    plt.show()
    
def global_E(data):
    n = len(data)
    N = np.shape(data[1][1][0,:])[0]
    Es = np.zeros([2,n,N])
    x = []
    for k, v in sorted(data.items()):
            x.append(k)
    j = 0
    for k, v in sorted(data.items()):
        for i in range(2):
            Es[i,j,:] = v[1][i,:]
        j+=1
    E_gl = np.mean(Es, axis=2)
    E_std = np.std(Es, axis=2)
    plt.plot(x, E_gl[0,:], label='E_12', c='orange')
    plt.fill_between(x, E_gl[0,:]-E_std[0,:], E_gl[0,:]+E_std[0,:], facecolor='orange', alpha=0.2)
    plt.plot(x, E_gl[1,:], label='E_13', c='red')
    plt.fill_between(x, E_gl[1,:]-E_std[1,:], E_gl[1,:]+E_std[1,:], facecolor='red', alpha=0.2)
    plt.legend()
    plt.xlabel('$R_2$')
    plt.ylabel('Global error $E_{ij}$')
    plt.title('Global mean synchronization error with std for changing $R_2$')
    plt.show()

