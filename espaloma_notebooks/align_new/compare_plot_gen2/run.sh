#BSUB -W 1:00
#BSUB -R "rusage[mem=10]"
#BSUB -J "qual[1-1000]"

python run.py $LSB_JOBINDEX
