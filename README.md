# Heat Equation in 3D

Its solution is

$ \widehat u(t,k) = \int_0^t e^{-4\pi^2 \nu k^2(t-s)} \widehat f(s,k) ds $

Its variance, in the stationary state, under homogeneous and delta-in-time forcing, is

$ \mathbb{E}[u^2] = \int d^3k \int_0^t e^{-4\pi^2 \nu k^2 (t-s)} \widehat C_f(k) $

which in the stationary state is

$ \int d^3k \frac{1}{4 \pi \nu k^2} \widehat C_f(k) $

and the velocity gradient, in the stationary state, is

$ \int d^3k \frac{1}{3\nu} \widehat C_f(k) $

$1/3$ appears because this is isotropic

for Gaussian correlation functions, the velocity variance is $L^2/2\nu$ and the velocity gradient variance is $1/6\nu$

`qsub ./sub_mkl_mpi.sh 0 5 5 0.1 1.0`

Real scalar heat equation 3D

N=7 NT=5 np=32
Wallclock time   = 00:45:06
Max vmem         = 1.098G

N=6 NT=6 np=32
Wallclock Time   = 00:46:11
Max vmem         = 1.721G
