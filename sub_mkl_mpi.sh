#!/bin/bash
#
### variables SGE
### shell du job
#$ -S /bin/bash
### nom du job (a changer)
#$ -N Heat3D
### file d'attente (a changer)
#$ -q h6-E5-2667v4deb128
### parallel environment & nb cpu (NSLOTS)
#$ -pe mpi16_debian 32
### charger l'environnement utilisateur pour SGE
#$ -cwd
### exporter les variables d'environnement sur tous les noeuds d'execution
#$ -V
### mails en debut et fin d'execution
#$ -m be

if [ "$#" -ne 5 ]; then
    echo "Five parameters needed: ensemble id, BN, BNT, nu, f0"
    exit 2
fi

# les files d'attente
# http://www.ens-lyon.fr/PSMN/doku.php?id=documentation:clusters:batch#les_files_d_attente

# donné par le système de batch
HOSTFILE=${TMPDIR}/machines

# aller dans le repertoire de travail/soumission
# important, sinon, le programme est lancé depuis ~/
cd ${SGE_O_WORKDIR}

# init env (should be in ~/.profile)
source /usr/share/lmod/lmod/init/bash

### configurer l'environnement
module purge
module load GCC/7.2.0 GCC/7.2.0/OpenMPI/3.0.0 Intel+MKL/2017.4
#export OMP_NUM_THREADS=8

### au besoin, forcer l'env OpenMPI
PREFIX="/applis/PSMN/debian9/software/Compiler/GCC/7.2.0/OpenMPI/3.0.0/"
MPIRUN=${PREFIX}/bin/mpirun

### execution du programme
${MPIRUN} -v -prefix ${PREFIX} -hostfile ${HOSTFILE} -np ${NSLOTS} ./heat_mpi.x $1 $2 $3 $4 $5 $6

# fin
