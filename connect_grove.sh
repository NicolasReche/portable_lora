#!/bin/bash
# Script rapide pour se connecter au serveur Grove

knock -v -d 700 134.226.42.250 49191 60998 65159 58436
ssh nreche@grove.computing.dcu.ie -J nreche@lir.adaptcentre.ie:8443

# === MEMO DES COMMANDES SLURM (À taper UNE FOIS CONNECTÉ sur Grove) ===
# sbatch jobs/sft_train.job          -> Pour lancer un entraînement
# squeue -u nreche                   -> Pour voir si le job est en attente (PD) ou tourne (R)
# cat resultats_entrainement.log     -> Pour lire tout le fichier de log (le "cat truc" !)
# tail -f resultats_entrainement.log -> Pour lire les logs en direct (Ctrl+C pour quitter)
# scancel LE_NUMERO_DU_JOB           -> Pour annuler un entraînement
