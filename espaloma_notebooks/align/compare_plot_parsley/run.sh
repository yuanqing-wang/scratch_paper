for idx in {0..1000}
do
    bsub -W 1:00 -R "rusage[mem=10]" python run.py $idx
done
