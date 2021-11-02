/*

(local)

make movecomp2

(psmn)

module load GCC/7.2.0 GCC/7.2.0/OpenMPI/3.0.0 Intel+MKL/2017.4
make heat_mpi

// one process

qsub ./sub_mkl_mpi.sh 1 6 5 0.01 1.0

// ensemble of processes, same parameters
seq 0 5 | xargs -I{} -P 6 qsub sub_mkl_mpi.sh {}

// many processes with different parameters
bash run_ccube_mpi.sh

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include "mkl_vsl.h"

#define error(x)      {printf("\n\nError generating,creating or opening "x"\n\n");exit(-1);}
#define errorrc(x)    {printf("\n\nError reading %s\nMaybe file does not exist\n\n",x);exit(-1);}
#define errorwc(x)    {printf("\n\nError generating,creating or writing %s\n\n",x);exit(-1);}
#define CLOSEFILE(x)  {fclose(x); x = NULL;}
#define SQR(x)        ((x)*(x))
#define FREEP(x)      {free(x); x = NULL;}
#define sfsg          {printf("\n\n So far, so good...");getchar();printf("\n\n");}
#define ABS2(i,j,k)   ( SQR(((double)(i))) + SQR(((double)(j))) + SQR(((double)(k))) )
#define CRDI(i,j,k)		(N*HN*(i)+HN*(j)+k) // coordinates in 3D complex array
#define CRDR(i,j,k)   (N*DN*(i)+DN*(j)+k) // coordinates in 3D real array (DN due to MPI assignment)

// MKL RNG
#define BRNG    VSL_BRNG_MCG31
#define METHOD  VSL_RNG_METHOD_GAUSSIAN_ICDF

// k is the fastest axis, with size N3
// j is the middle axis, with size N2
// i is the slowest axis, with size N1
// CRDI(i,j,k) = N2 N3 i + N3 j + k

/****** global variables ******/

typedef long int LI;
typedef unsigned long int ULI;
static const long double TWOPI =  6.2831853071795864769252867665590058L;
static const long double PISQR =  9.8696044010893586188344909998761511L;
fftw_plan plan_fx_f, plan_fx_b, plan_fy_f, plan_fy_b, plan_fz_f, plan_fz_b;
fftw_plan plan_ux_f, plan_ux_b, plan_uy_f, plan_uy_b, plan_uz_f, plan_uz_b;
// MPI variables
ptrdiff_t alloc_local, local_n0, local_0_start;
int id,np;
// Code parameters
LI N,pid,numsteps,SEED;
double nu,f0;

/****** functions ******/
double gauss_kernel(double k, double PIL2);
static inline void write_complex_3D_array(fftw_complex *y, LI pid, LI N,
	LI it, double L, char axis);
static inline void write_real_1D_array(double *y, LI pid, LI N, LI numsteps,
	double L,	char axis);
static inline void jentzen_kloeden_winkel_step(
	fftw_complex *ukx,
	fftw_complex *gx,
	fftw_complex *tx,
	double *K, double *K2, LI id, double dt, double sqdx, double visc);
static inline void gen_force3D(double *fx, double *fy, double *fz,
	fftw_complex *gx, fftw_complex *gy, fftw_complex *gz, double *ker,
	LI N, LI HN, LI DN,
	double TPI3, double PIL2, double sqdx,
	VSLStreamStatePtr stream, double *rands);
static inline void euler_maruyama_step(fftw_complex *ukx, fftw_complex *gx,
	double *K2, LI N, LI HN, double dt, double sqdt, double visc);
static inline void predictor_corrector_step(fftw_complex *ukx, fftw_complex *gx,
	fftw_complex *tx,
	double *K2, LI N, LI HN, double dt, double sqdt, double visc);

int main(int argc, char **argv){

	LI it, HN, DN;
	LI i,j,k;
  //extern long double TWOPI, PISQR;
	double TPI3, PIL2;
	// f and g are Fourier transform pairs
	// f is the external force, in real space
	// u and v are Fourier transform pairs
	// u is the velocity vector, in real space
	// t is a temp array, in Fourier space, used in predictor-corrector algorithm
	double *K, *K2, *ker, *rands;
	double *var_f, *var_x, *var_d1, *var_d2, *var_d3, *tmp; // observables
	fftw_complex *gx, *gy, *gz, *ukx, *uky, *ukz, *tx, *ty, *tz; /* arrays */
	//int dim;
	double dx,sqdx,Ltot,L,dt,sqdt,visc,normN3,dtcte;
	VSLStreamStatePtr stream; // MKL RNG
	extern ptrdiff_t alloc_local, local_n0, local_0_start;
	extern LI N,pid,SEED,numsteps;
	extern int id,np;
	extern double nu,f0;

	// Grid size
	N = (LI) 1<<atoi(argv[2]); // 1<<N = 2^N
	HN = (int)(N/2)+1; // half-N
	DN = 2*HN; // double-N
	numsteps = (LI) pow(10,atoi(argv[3]));

	Ltot = 1.;
	L = .1*Ltot;
	dx = Ltot/(double)N;
	sqdx = dx * sqrt(dx); // StDev(dW_x) = dx^{dim/2}
	nu = atof(argv[4]);
	f0 = atof(argv[5]); // forcing amplitude

	// Simulation time
	// Time resolution must be roughly
	// dt = 0.1 dx^2 / (3 * pi^2 * nu * Ltot^2)
	// So that every Fourier mode is well resolved
	//dt = .1*dx*dx/(3.*PISQR*nu*Ltot*Ltot);
	dtcte = .1;
	dt = dtcte*dx*dx;
	sqdt = sqrt(dt);
	visc = 4.*PISQR*nu;
	normN3 = 1./((double)(N*N*N));

	// size of time steps vs. dx^2
	//printf("dt = %.3f dx^2\n",dt/dx/dx);

	MPI_Init( &argc, &argv );
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	fftw_mpi_init();

	pid = atoi(argv[1]); // process id, for ensemble average
	/* Intializes random number generator with different seed for each process */
	SEED = 12345 + pid*np + id;

	if(id==0){

		for(i=0;i<argc;i++){
			printf("%s ",argv[i]);
		}
		printf("\n\n");

		printf("Simulation parameters \n\n N=2^%02d \n numsteps=10^%02d \n L=%.03f \n nu=%.03e \
		\n f0=%.03e \n dt=%.03e*dx^2\n\n",(int)(log2(N)),(int)(log10(numsteps)),L,nu,f0,dtcte);

	}

	// initialize RNG
  vslNewStream( &stream, BRNG,  SEED );

	// Allocating necessary arrays
	alloc_local = fftw_mpi_local_size_3d(N, N, HN, MPI_COMM_WORLD, &local_n0, &local_0_start);
	// Fourier space, alloc_local = local_n0 * N * HN
	// real space, the arrays have size 2*alloc_local

	if( (K = (double*) malloc(sizeof(double) * N)) == NULL)
		error("vector K");
	if( (K2 = (double*) malloc(sizeof(double) * 2 * alloc_local)) == NULL)
		error("vector K2");
	if( (rands = (double*) malloc(sizeof(double) * 2 * alloc_local)) == NULL)
		error("vector rands");
	if( (ker = (double*) malloc(sizeof(double) * N)) == NULL)
		error("vector ker");
	if( (gx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local )) == NULL)
		error("vector gx");
	if( (gy = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local )) == NULL)
		error("vector gy");
	if( (gz = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local )) == NULL)
		error("vector gz");
	if( (ukx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local )) == NULL)
		error("vector ukx");
	if( (uky = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local )) == NULL)
		error("vector uky");
	if( (ukz = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local )) == NULL)
		error("vector ukz");
	if( (tx = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local)) == NULL)
		error("vector tx");
	if( (ty = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local)) == NULL)
		error("vector ty");
	if( (tz = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * alloc_local)) == NULL)
		error("vector tz");

	double *fx = (double *) gx;
	double *fy = (double *) gy;
	double *fz = (double *) gz;
	double *uxx = (double *) ukx;
	double *uxy = (double *) uky;
	double *uxz = (double *) ukz;

	if( (var_f = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector var_f");
	if( (var_x = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector var_x");
	if( (var_d1 = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector var_d1");
	if( (var_d2 = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector var_d2");
	if( (var_d3 = (double*) malloc(sizeof(double) * numsteps)) == NULL)
		error("vector var_d3");
	if(id==0){
		if( (tmp = (double*) malloc(sizeof(double) * numsteps)) == NULL)
			error("vector tmp");
	}

	/** initialize FFTW **/
	// Force vector transforms
	plan_fx_f = fftw_mpi_plan_dft_r2c_3d(N, N, N, fx, gx, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_fx_b = fftw_mpi_plan_dft_c2r_3d(N, N, N, gx, fx, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_fy_f = fftw_mpi_plan_dft_r2c_3d(N, N, N, fy, gy, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_fy_b = fftw_mpi_plan_dft_c2r_3d(N, N, N, gy, fy, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_fz_f = fftw_mpi_plan_dft_r2c_3d(N, N, N, fz, gz, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_fz_b = fftw_mpi_plan_dft_c2r_3d(N, N, N, gz, fz, MPI_COMM_WORLD, FFTW_MEASURE);
	// Velocity vector transforms
	plan_ux_f = fftw_mpi_plan_dft_r2c_3d(N, N, N, uxx, ukx, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_ux_b = fftw_mpi_plan_dft_c2r_3d(N, N, N, ukx, uxx, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_uy_f = fftw_mpi_plan_dft_r2c_3d(N, N, N, uxy, uky, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_uy_b = fftw_mpi_plan_dft_c2r_3d(N, N, N, uky, uxy, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_uz_f = fftw_mpi_plan_dft_r2c_3d(N, N, N, uxz, ukz, MPI_COMM_WORLD, FFTW_MEASURE);
	plan_uz_b = fftw_mpi_plan_dft_c2r_3d(N, N, N, ukz, uxz, MPI_COMM_WORLD, FFTW_MEASURE);

	// Array of frequencies in Fourier space
  K[0]=0.0;
  K[N/2]=(double)(N/2)/Ltot;
	for(i=1;i<N/2;i++){
	  K[i]=(double)i/Ltot;
	  K[N-i]=-(double)i/Ltot;
  }

	// 3d array, absolute value of frequency in Fourier space
	for(i=0;i<local_n0;i++){
		for(j=0;j<N;j++){
			for(k=0;k<HN;k++){
				K2[CRDI(i,j,k)]=SQR(K[local_0_start+i])+SQR(K[j])+SQR(K[k]);
			}
		}
  }

	/* correlation function of external force, at large scales
	   Fourier transform convention: {0,-2 Pi} (Mathematica)
	   Cf(x) = exp(-x^2/(2 L^2))
	   kernel = f_hat(k) = sqrt( F[ Cf(x) ] )
		 f_hat(k) = sqrt(sqrt(2 pi)) * sqrt(L) * exp(-pi^2 L^2 k^2)
	*/
	TPI3 = pow(TWOPI,0.75)*pow(L,1.5); // (2 pi)^(3/4) * L^(3/2)
	PIL2 = PISQR*L*L; // pi^2*L^2

	// Assign kernel operator directly in Fourier space
	// Imag. components are zero
	// frequencies are i/Ltot, normalization is eps = 1/Ltot
	for(i=0;i<N;i++){
		ker[i] = gauss_kernel(K[i],PIL2);
	}

	// set initial condition in Fourier space, v=0
	for(i=0;i<alloc_local;i++){
		ukx[i] = 0.;
	}
	for(i=0;i<alloc_local;i++){
		uky[i] = 0.;
	}
	for(i=0;i<alloc_local;i++){
		ukz[i] = 0.;
	}

	for(it=0;it<numsteps;it++){
		var_f[it] = 0.;
	}
	for(it=0;it<numsteps;it++){
		var_x[it] = 0.;
	}
	for(it=0;it<numsteps;it++){
		var_d1[it] = 0.;
	}
	for(it=0;it<numsteps;it++){
		var_d2[it] = 0.;
	}
	for(it=0;it<numsteps;it++){
		var_d3[it] = 0.;
	}

	for(it=0;it<numsteps;it++){

	  gen_force3D(fx,fy,fz,gx,gy,gz,ker,N,HN,DN,TPI3,PIL2,sqdx,stream,rands);

		//euler_maruyama_step(ukx,gx,K2,N,HN,dt,sqdt,visc);
		predictor_corrector_step(ukx,gx,tx,K2,N,HN,dt,sqdt,visc);
	  //jentzen_kloeden_winkel_step(ukx,gx,tx,K,K2,id,dt,sqdx,visc);

	  // Sums of variances of Fourier modes
		for(i=0;i<alloc_local;i++){
	    var_f[it] += SQR(cabs(ukx[i]));
	  }
	  // remove zero mode only
	  if(id==0){
	    var_f[it] -= SQR(cabs(ukx[0]));
	  }

		// variance of velocity gradient in x direction
	  // we can reuse g/f and its transform for that

	  // Begins Var(dx uxx)
	  for(i=0;i<local_n0;i++){
	    for(j=0;j<N;j++){
	      for(k=0;k<HN;k++){
	        gx[CRDI(i,j,k)] = I*TWOPI*K[local_0_start+i]*ukx[CRDI(i,j,k)];
	      }
	    }
	  }

	  MPI_Barrier(MPI_COMM_WORLD);
	  // gradient back to real space
	  fftw_execute(plan_fx_b);

	  // save values, variance of velocity gradient
	  // ijk loops are needed because of padding in real space
	  for(i=0;i<local_n0;i++){
	    for(j=0;j<N;j++){
	      for(k=0;k<N;k++){
	        var_d1[it] += SQR(fx[CRDR(i,j,k)]);
	      }
	    }
	  }
	  // End of Var(dx uxx)

		// Begins Var(dy uxy)
	  for(i=0;i<local_n0;i++){
	    for(j=0;j<N;j++){
	      for(k=0;k<HN;k++){
	        gx[CRDI(i,j,k)] = I*TWOPI*K[j]*ukx[CRDI(i,j,k)];
	      }
	    }
	  }


	  MPI_Barrier(MPI_COMM_WORLD);
	  // gradient back to real space
	  fftw_execute(plan_fx_b);

	  // save values, variance of velocity gradient
	  // ijk loops are needed because of padding in real space
	  for(i=0;i<local_n0;i++){
	    for(j=0;j<N;j++){
	      for(k=0;k<N;k++){
	        var_d2[it] += SQR(fx[CRDR(i,j,k)]);
	      }
	    }
	  }
	  // End of Var(dy uxy)

	  // Begins Var(dz uxy)
	  for(i=0;i<local_n0;i++){
	    for(j=0;j<N;j++){
	      for(k=0;k<HN;k++){
	        gx[CRDI(i,j,k)] = I*TWOPI*K[k]*ukx[CRDI(i,j,k)];
	      }
	    }
	  }


	  MPI_Barrier(MPI_COMM_WORLD);
	  // gradient back to real space
	  fftw_execute(plan_fx_b);

	  // save values, variance of velocity gradient
	  // ijk loops are needed because of padding in real space
	  for(i=0;i<local_n0;i++){
	    for(j=0;j<N;j++){
	      for(k=0;k<N;k++){
	        var_d3[it] += SQR(fx[CRDR(i,j,k)]);
	      }
	    }
	  }
	  // End of Var(dz uxy)

	  // backup velocities in Fourier space
	  for(i=0;i<alloc_local;i++)
	    tx[i] = ukx[i];
	  for(i=0;i<alloc_local;i++)
	    ty[i] = uky[i];
	  for(i=0;i<alloc_local;i++)
	    tz[i] = ukz[i];

	  // Wait for all processes to sync
	  MPI_Barrier(MPI_COMM_WORLD);

	  // velocities back to real space
	  fftw_execute(plan_ux_b);
	  fftw_execute(plan_uy_b);
	  fftw_execute(plan_uz_b);

	  for(i=0;i<local_n0;i++){
	    for(j=0;j<N;j++){
	      for(k=0;k<N;k++){
	        var_x[it] += SQR(uxx[CRDR(i,j,k)]);
	      }
	    }
	  }

	  // bring back velocities in Fourier space from backup
	  for(i=0;i<alloc_local;i++)
	    ukx[i] = tx[i];
	  for(i=0;i<alloc_local;i++)
	    uky[i] = ty[i];
	  for(i=0;i<alloc_local;i++)
	    ukz[i] = tz[i];

	  /*// print velocity field at some of the steps
	  write_complex_3D_array(uxx,pid,N,alloc_local,it,L,nu,f0,'x');
	  write_complex_3D_array(uxy,pid,N,alloc_local,it,L,nu,f0,'y');
	  write_complex_3D_array(uxz,pid,N,alloc_local,it,L,nu,f0,'z');*/

	}

	// spatial average variance velocity
	for(it=0;it<numsteps;it++){
	  var_x[it] *= normN3;
	}
	// spatial average variance velocity gradient
	for(it=0;it<numsteps;it++){
	  var_d1[it] *= normN3;
	}
	// spatial average variance velocity gradient
	for(it=0;it<numsteps;it++){
	  var_d2[it] *= normN3;
	}
	// spatial average variance velocity gradient
	for(it=0;it<numsteps;it++){
	  var_d3[it] *= normN3;
	}

	// sum all variances (v, Fourier space) into arrays on root process
	MPI_Reduce(var_f,tmp,numsteps,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(id==0){
	  write_real_1D_array(tmp,pid,N,numsteps,L,'f');
	}

	// sum all variances (u, real space) into arrays on root process
	MPI_Reduce(var_x,tmp,numsteps,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(id==0){
	  write_real_1D_array(tmp,pid,N,numsteps,L,'x');
	}

	// sum all variances (v, Fourier space) into arrays on root process
	MPI_Reduce(var_d1,tmp,numsteps,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(id==0){
	  write_real_1D_array(tmp,pid,N,numsteps,L,'1');
	}

	// sum all variances (v, Fourier space) into arrays on root process
	MPI_Reduce(var_d2,tmp,numsteps,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(id==0){
	  write_real_1D_array(tmp,pid,N,numsteps,L,'2');
	}

	// sum all variances (v, Fourier space) into arrays on root process
	MPI_Reduce(var_d3,tmp,numsteps,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
	if(id==0){
	  write_real_1D_array(tmp,pid,N,numsteps,L,'3');
	}

	/***********

	CLOSING STAGE

	***********/

	vslDeleteStream( &stream ); // close MKL RNG

  fftw_destroy_plan(plan_fx_f);
  fftw_destroy_plan(plan_fx_b);
	fftw_destroy_plan(plan_fy_f);
  fftw_destroy_plan(plan_fy_b);
	fftw_destroy_plan(plan_fz_f);
  fftw_destroy_plan(plan_fz_b);
	fftw_destroy_plan(plan_ux_f);
  fftw_destroy_plan(plan_ux_b);
	fftw_destroy_plan(plan_uy_f);
  fftw_destroy_plan(plan_uy_b);
	fftw_destroy_plan(plan_uz_f);
  fftw_destroy_plan(plan_uz_b);
	fftw_free(gx);
	fftw_free(gy);
	fftw_free(gz);
	fftw_free(ukx);
	fftw_free(uky);
	fftw_free(ukz);
	fftw_free(tx);
	fftw_free(ty);
	fftw_free(tz);
	FREEP(ker);
	FREEP(K);
	FREEP(K2);
	FREEP(rands);

	FREEP(var_f);
	FREEP(var_x);
	FREEP(var_d1);
	FREEP(var_d2);
	FREEP(var_d3);
	if(id==0)
		FREEP(tmp);

	if(id==0)
		printf("\n\n And we are done \n\n");

	fftw_mpi_cleanup();
	MPI_Finalize();

return 0;
}

double gauss_kernel(double k, double PIL2){
	return exp(-PIL2*k*k);
}

static inline void write_complex_3D_array(fftw_complex *y, LI pid, LI N,
	LI it, double L, char axis){

  char name[200];
  FILE *fout;
	int BN = (int)log2(N);
	int j;

	// sum all variances (u, real space) into arrays on root process
	if(id==0){

		// open new file. if already existing, erase it
		sprintf(name,"data/HeatF_%c_R_%04ld_N_%02d_IT_%06ld_L_%.3e_nu_%.3e_f0_%.3e.dat",axis,pid,BN,it,L,nu,f0);
	  if( (fout = fopen(name,"w")) == NULL)
	    errorwc(name);

		// write velocity array saved on rank 0
		fwrite(y, sizeof(y[0]), alloc_local, fout);

		CLOSEFILE(fout);

		// open file again, for appending
		sprintf(name,"data/HeatF_%c_R_%04ld_N_%02d_IT_%06ld_L_%.3e_nu_%.3e_f0_%.3e.dat",axis,pid,BN,it,L,nu,f0);
	  if( (fout = fopen(name,"a")) == NULL)
	    errorwc(name);

		// Receive data from all other processes
		for(j=1;j<np;j++){

				MPI_Recv(y, alloc_local, MPI_DOUBLE_COMPLEX, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

				fwrite(y, sizeof(y[0]), alloc_local, fout);

		}

		CLOSEFILE(fout);

	} else {

		// Each process !=0 sends data to root
		MPI_Send(y, alloc_local, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);

	}

}

static inline void write_real_1D_array(double *y, LI pid, LI N, LI numsteps,
	double L,	char axis){

  char name[200];
  FILE *fout;
	int BN = (int)log2(N);
	int BNT = (int)log10(numsteps);
	extern double nu,f0;

	sprintf(name,"data/HeatVar_%c_R_%04ld_N_%02d_NT_%02d_L_%.3e_nu_%.3e_f0_%.3e.dat",axis,pid,BN,BNT,L,nu,f0);
  if( (fout = fopen(name,"w")) == NULL)
    errorwc(name);

  // binary output
  fwrite(y, sizeof(y[0]), numsteps, fout);

	CLOSEFILE(fout);

}

static inline void gen_force3D(double *fx, double *fy, double *fz,
	fftw_complex *gx, fftw_complex *gy, fftw_complex *gz, double *ker,
	LI N, LI HN, LI DN,
	double TPI3, double PIL2, double sqdx,
	VSLStreamStatePtr stream, double *rands){

	ptrdiff_t i,j,k;
	double cte1,cte2,cte3,norm;
	double m1=0.,m2=1.;

	// call RNG
	vdRngGaussian( METHOD, stream, 2*alloc_local, rands, m1, m2 );
	// Assign random vector dW (white noise increments)
	for(i=0;i<local_n0;i++){
		for(j=0;j<N;j++){
			for(k=0;k<N;k++){
				fx[CRDR(i,j,k)] = rands[CRDR(i,j,k)] * sqdx;
			}
		}
	}

	fftw_execute(plan_fx_f);

	// Tried to save memory, ended up with a large stride
	// Let's make it work, later I have to see what is best
	// I have to do this at every step, hence optimization will be
	// important for the program overall
	for(i=0;i<local_n0;i++){
		cte1 = TPI3 * ker[local_0_start+i];
		for(j=0;j<N;j++){
			cte2 = ker[j];
			for(k=0;k<HN;k++){
				cte3 = ker[k];
				norm = cte1 * cte2 * cte3;
				gx[CRDI(i,j,k)] = gx[CRDI(i,j,k)] * norm;
			}
		}
	}

}

static inline void euler_maruyama_step(fftw_complex *ukx, fftw_complex *gx,
	double *K2, LI N, LI HN, double dt, double sqdt, double visc){

	ptrdiff_t i;
	double viscdt = visc*dt;
	double f0sqdt = sqrt(f0*dt);

	// deterministic evolution
	for(i=0;i<alloc_local;i++){
		ukx[i] -= viscdt * K2[i] * ukx[i];
	}

	// add stochastic force
	for(i=0;i<alloc_local;i++){
		ukx[i] += f0sqdt * gx[i];
	}

}

static inline void predictor_corrector_step(fftw_complex *ukx, fftw_complex *gx,
	fftw_complex *tx,
	double *K2, LI N, LI HN, double dt, double sqdt, double visc){
// order 1.0 predictor corrector algorithm
// see Kloeden-Platen p. 502

	ptrdiff_t i;
	double viscdt = visc*dt;
	double f0sqdt = sqrt(f0*dt);

	// predictor step
	// t* are temp arrays, to store predictor array

	// initial setup of predictor step
	for(i=0;i<alloc_local;i++){
		tx[i] = ukx[i];
	}

	// deterministic evolution
	for(i=0;i<alloc_local;i++){
		tx[i] -= viscdt * K2[i] * ukx[i];
	}

	// add stochastic force
	for(i=0;i<alloc_local;i++){
		tx[i] += f0sqdt * gx[i];
	}

	// corrector step

	// deterministic evolution, half step using last velocity array
	for(i=0;i<alloc_local;i++){
		ukx[i] -= .5 * viscdt * K2[i] * ukx[i];
	}

	// deterministic evolution, half step using predictor array
	for(i=0;i<alloc_local;i++){
		ukx[i] -= .5 * viscdt * K2[i] * tx[i];
	}

	// add stochastic force
	for(i=0;i<alloc_local;i++){
		ukx[i] += f0sqdt * gx[i];
	}

}


// Jentzen, Kloeden and Winkel, Annals of Applied Probability 21.3 (2011): 908-950
// see eq. 21
static inline void jentzen_kloeden_winkel_step(
	fftw_complex *ukx,
	fftw_complex *gx,
	fftw_complex *tx,
	double *K, double *K2, LI id, double dt, double sqdx, double visc){

	LI i, i0;
	double cte;

	// step for x component of vectors

	// zero mode is only present in rank 0
	if(id==0){
		// zero mode
		cte    = sqrt(dt*f0);
		// stochastic part
		ukx[0] += cte * gx[0];
		i0 = 1;
	} else {
		i0 = 0;
	}

	cte = dt*visc;
	// deterministic part
	for(i=i0;i<alloc_local;i++){
		ukx[i] *= exp(-cte*K2[i]);
	}
	// stochastic part
	cte = sqrt(.5*f0/visc);
	for(i=i0;i<alloc_local;i++){
		tx[i]  = cte * gx[i];
	}
	cte = 2.*visc*dt;
	for(i=i0;i<alloc_local;i++){
		tx[i] *= sqrt((1.-exp(-cte*K2[i]))/K2[i]);
	}
	for(i=i0;i<alloc_local;i++){
		// stochastic part
		ukx[i] += tx[i];
	}

}
