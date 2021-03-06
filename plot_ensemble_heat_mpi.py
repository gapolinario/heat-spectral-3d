# (local) $ make movecomp2
# (psmn)  $ module load Python/3.6.1
# (psmn)  $ python3 plot_ensemble_heat_mpi.py

import numpy as np
import matplotlib.pyplot as plt

# external parameters

R = 0
BN = 6
BNT = 6
Lcte = .1
nu = .2
f0 = 1.
size=12

N = 2**BN
numsteps=10**BNT
Ltot = 1.
dx = Ltot/N
L = Lcte*Ltot
X = np.fft.fftfreq(N) * Ltot
K = np.fft.fftfreq(N) * N / Ltot
Cf = np.exp(-.5*X*X/L/L)
N2 = N//2+1
visc = 4.*np.pi*np.pi*nu

#dt = .1*dx*dx/(3.*np.pi*np.pi*nu*Ltot*Ltot)
dtcte = .1
dt = dtcte*dx*dx










################################################################################
################################################################################
################################################################################
################################################################################
############################    THEORY PART    #################################















# correlation function of force in Fourier space
# E[g(k)g^*(k')]
# g = F[f]
def cfk_theo(x,y,z):
    return np.power(2.*np.pi,1.5)*L*L*L*np.exp(-2.*np.pi*np.pi*L*L*(K[int(x)]**2+K[int(y)]**2+K[int(z)]**2))

# correlation function of velocity field in Fourier space
# E[v(k)v^*(k')]
# v = F[u]
def cuk_theo(x,y,z):
    # this is for ornstein-uhlenbeck
    #return .5*f0/visc*cfk_theo(kx,ky,kz)
    # now for 1d heat equation
    return .5*f0*cfk_theo(x,y,z)/visc/(K[int(x)]**2+K[int(y)]**2+K[int(z)]**2)#load velocity fields for one realization

# correlation function of zero mode of velocity field in Fourier space
# E[v(0)v^*(0)]
def cuk_zero(int_time):
    return f0*cfk_theo(0,0,0)*int_time*dt

#"""
# lines below, verifying sums of variances of Fourier modes
# excludes Fourier modes with ki=0
var_theo  = np.sum(np.fromiter((cuk_theo(i,j,k) for i in range(1,N) for j in range(1,N) for k in range(1,N2)), float))
# Fourier modes with one ki=0
var_theo += np.sum(np.fromiter((cuk_theo(0,j,k) for j in range(1,N) for k in range(1,N2)), float))
var_theo += np.sum(np.fromiter((cuk_theo(i,0,k) for i in range(1,N) for k in range(1,N2)), float))
var_theo += np.sum(np.fromiter((cuk_theo(i,j,0) for i in range(1,N) for j in range(1,N)), float))
# Fourier modes with two ki=kj=0,i!=j
var_theo += np.sum(np.fromiter((cuk_theo(0,0,k) for k in range(1,N2)), float))
var_theo += np.sum(np.fromiter((cuk_theo(0,i,0) for i in range(1,N) ), float))
var_theo += np.sum(np.fromiter((cuk_theo(i,0,0) for i in range(1,N) ), float))
# Zero mode is added separately, because it depends on time
#"""












################################################################################
################################################################################
################################################################################
################################################################################
#########################    FOURIER SPACE VAR    ##############################
























var_time = np.zeros(numsteps)
for i in range(size):
    var_time += np.fromfile("data/HeatVar_f_R_{:04d}_N_{:02d}_NT_{:02d}_L_{:.3e}_nu_{:.3e}_f0_{:.3e}.dat".format(i,BN,BNT,L,nu,f0),dtype=np.double)
var_time *= 1./size

fig, axs = plt.subplots(3,2,figsize=(14,10))

# check variance of all fourier modes (but for zero mode)
axs[0,0].axhline(var_theo,color='k',linestyle='dashed')
axs[0,0].axhline(np.mean(var_time[numsteps//2:]),color='red')
axs[0,0].plot(var_time)
axs[0,0].set_xlabel(r'$t$')
axs[0,0].set_ylabel(r'$\sum_k \mathbb{E}[|\widehat u_k|^2]$')

"""
# check total variance (discrete sum)
plt.plot(var)
plt.plot(var_theo+cuk_zero(range(numsteps)),color='grey',linestyle='dashed')
plt.show()
"""















################################################################################
################################################################################
################################################################################
################################################################################
#########################    REAL SPACE VAR    #################################


















##### VARIANCE

# theoretical value for variance in real space
var_theo = .5*f0*L*L/nu

var_time = np.zeros(numsteps)
for i in range(size):
    var_time += np.fromfile("data/HeatVar_x_"+
    "R_{:04d}_N_{:02d}_NT_{:02d}_L_{:.3e}_nu_{:.3e}_f0_{:.3e}".format(i,BN,BNT,L,nu,f0)
    +".dat",dtype=np.double)
var_time *= 1./size

# plot full time evolution
#"""
axs[0,1].plot(var_time,color='#7fc97f') # full time evolution
axs[0,1].axhline(np.mean(var_time[numsteps//2:]),color='red') # numerical mean
axs[0,1].axhline(y=var_theo,color='grey',linestyle='dashed')  # theoretical value
axs[0,1].set_xlabel(r'$t$')
axs[0,1].set_ylabel(r'$\mathbb{E}[|u|^2]$')

##### GRADIENT VARIANCE d_x u_x


# theoretical value for variance in real space
# one direction, 1/6, sum of all 3 directions would give 1/2
var_theo = f0/6./nu

var_time = np.zeros(numsteps)
for i in range(size):
    var_time += np.fromfile("data/HeatVar_1_"+
    "R_{:04d}_N_{:02d}_NT_{:02d}_L_{:.3e}_nu_{:.3e}_f0_{:.3e}".format(i,BN,BNT,L,nu,f0)
    +".dat",dtype=np.double)
var_time *= 1./size

axs[1,0].plot(var_time,color='#7fc97f') # full time evolution
axs[1,0].axhline(np.mean(var_time[numsteps//2:]),color='red') # numerical mean
axs[1,0].axhline(y=var_theo,color='grey',linestyle='dashed')  # theoretical value
axs[1,0].set_xlabel(r'$t$')
axs[1,0].set_ylabel(r'$\mathbb{E}[(\partial_x u)^2]$')


##### GRADIENT VARIANCE d_z u_y


# theoretical value for variance in real space
# one direction, 1/6, sum of all 3 directions would give 1/2
var_theo = f0/6./nu

var_time = np.zeros(numsteps)
for i in range(size):
    var_time += np.fromfile("data/HeatVar_2_"+
    "R_{:04d}_N_{:02d}_NT_{:02d}_L_{:.3e}_nu_{:.3e}_f0_{:.3e}".format(i,BN,BNT,L,nu,f0)
    +".dat",dtype=np.double)
var_time *= 1./size

# plot full time evolution
#"""
axs[1,1].plot(var_time,color='#7fc97f') # full time evolution
axs[1,1].axhline(np.mean(var_time[numsteps//2:]),color='red') # numerical mean
axs[1,1].axhline(y=var_theo,color='grey',linestyle='dashed')  # theoretical value
axs[1,1].set_xlabel(r'$t$')
axs[1,1].set_ylabel(r'$\mathbb{E}[(\partial_y u)^2]$')


##### GRADIENT VARIANCE d_z u_y


# theoretical value for variance in real space
# one direction, 1/6, sum of all 3 directions would give 1/2
var_theo = f0/6./nu

var_time = np.zeros(numsteps)
for i in range(size):
    var_time += np.fromfile("data/HeatVar_3_"+
    "R_{:04d}_N_{:02d}_NT_{:02d}_L_{:.3e}_nu_{:.3e}_f0_{:.3e}".format(i,BN,BNT,L,nu,f0)
    +".dat",dtype=np.double)
var_time *= 1./size

axs[2,0].plot(var_time,color='#7fc97f') # full time evolution
axs[2,0].axhline(np.mean(var_time[numsteps//2:]),color='red') # numerical mean
axs[2,0].axhline(y=var_theo,color='grey',linestyle='dashed')  # theoretical value
axs[2,0].set_xlabel(r'$t$')
axs[2,0].set_ylabel(r'$\mathbb{E}[(\partial_z u)^2]$')


plt.tight_layout()
#plt.savefig("fgf1d.png")
plt.show()
