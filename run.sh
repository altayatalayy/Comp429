#!/bin/bash
#SBATCH -p short
#SBATCH -J udemircioglu17_hw2
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=hw2-%j.out  

 ~/hw2/cardiacSimGPU/cardiacsim -n 400 -t 500 -p 100