#!/bin/bash
# Quick script to connect to the Grove server

#knock -v -d 700 134.226.42.250 49191 60998 65159 58436
#ssh nreche@grove.computing.dcu.ie -J nreche@lir.adaptcentre.ie:8443

#knock -v -d 1000 134.226.42.250 49191 60998 65159 58436 && ssh -v -p 8443 nreche@lir.adaptcentre.ie
#ssh nreche@grove.computing.dcu.ie -J nreche@lir.adaptcentre.ie:8443

# === SLURM COMMANDS MEMO (To be typed ONCE CONNECTED on Grove) ===
# sbatch jobs/sft_train.job          -> To launch a training job
# squeue -u nreche                   -> To check if the job is pending (PD) or running (R)
# cat resultats_entrainement.log     -> To read the entire log file
# tail -f resultats_entrainement.log -> To read the logs live (Ctrl+C to quit)
# scancel JOB_ID                     -> To cancel a training job
