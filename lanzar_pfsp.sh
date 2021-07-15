#!/bin/bash
# Give your job a name, so you can recognize it in the queue overview
# Use at most 8 characters (only 8 are printed when the queue is shown)
SBATCH --job-name=BOPFSP

# Use this parameter to define the output and error files
# For job arrays, the %A represents the job ID and %a the array index
# You can add this files to a directory, but it has to exist (it does not create it)
#SBATCH --output=outputs/ManualMTL_%A_%a.out
#SBATCH --error=outputs/ManualMTL_%A_%a.err

# Number of cores to be used. This option uses only one core
SBATCH -n 1
# This option would use one node with all its cores
SBATCH -N 1
SBATCH --ntasks=1

# Remember to ask for enough memory for your process. The default is 2Gb
#SBATCH --mem-per-cpu=8000
# It is mandatory to indicate a maximum time for the process.
# If you have no idea about how much time it will take, set it at a big value
SBATCH --time=1000:00:00
# So far there is only one queue, compute
SBATCH --partition=2018allq
SBATCH -w, --nodelist=nodo31

# Determine the number of repetitions of the process
SBATCH --array=1-1

# Leave these options commented
# #SBATCH --cpus-per-task=1
# #SBATCH --threads-per-core=2
# #SBATCH --ntasks-per-core=2

#####################
# Here starts your code #
#####################

# Define and create a unique scratch directory for this job
SCRATCH_DIRECTORY=/var/tmp/${USER}/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
mkdir -p ${SCRATCH_DIRECTORY}/taillards_benchmark

cd ${SCRATCH_DIRECTORY}

# You can copy everything you need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was submitted from
cp ${SLURM_SUBMIT_DIR}/*.py ${SCRATCH_DIRECTORY}
cp ${SLURM_SUBMIT_DIR}/taillard_benchmark/*.fsp ${SCRATCH_DIRECTORY}/taillard_benchmark
#cp ${SLURM_SUBMIT_DIR}/random_benchmark/*.fsp ${SCRATCH_DIRECTORY}/random_benchmark

# This is where the actual work is done. In this case, the script only waits.
# You can use $SLURM_ARRAY_TASK_ID to differentiate the files of each repetition

source /home/djimenez/venvs/kalimero_venv/bin/activate
#echo "python3 kalimero_pfsp.py $1 $2 $3 $4" > output_$1_$2_$3_$4.txt
#for j in 20 50 100 #job
#do
#  for m in 5 10 20 #machine
#  do
#    for s in {1..10} #seed/instance
#    do
#      for a in $(seq 0.1 0.1 3.0)
#      do
#        for k in 0.00001 0.0001 0.001 0.1 1.0 10.0 100.0 1000.0 10000.0
#        do
#          python3 pfsp.py "taillard" $j $m $s $a $k 1 > "output_$1 _$j _$m _$s"
#          #python3 pfsp.py "taillard" $j $m $s $a $k 1 > "output_$1 _$j _$m _$s"
#        done
#      done
#    done
#  done
#done
python3 pfsp.py "taillard" 20 5 0 1 1 1
deactivate
echo DONE
# After the job is done we copy our output back to $SLURM_SUBMIT_DIR
#cp -r ${SCRATCH_DIRECTORY}/results* ${SLURM_SUBMIT_DIR}/results/
cp -r ${SCRATCH_DIRECTORY}/Output_$1_$2_$3_$4.txt ${SLURM_SUBMIT_DIR}/outputs/
cp -r ${SCRATCH_DIRECTORY}/results/* ${SLURM_SUBMIT_DIR}/results/
# After everything is saved to the home directory, delete the work directory
cd ${SLURM_SUBMIT_DIR}
rm -rf ${SCRATCH_DIRECTORY}
