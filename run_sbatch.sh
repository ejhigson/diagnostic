#!/usr/bin/env bash
# script runs an sbatch script on the python rootname input
#
# SBATCH settings
# start script
# ------------
if [ $# -eq 0 ]; then
		echo "No arguments supplied"
else
	python_file=$1
    # nodes=$2
    nodes=1
    runtime_hours=$2
    runtime_minutes=${3:-"00"}
fi
runtime="${runtime_hours}:${runtime_minutes}:00"
echo "Running ${python_file} with nodes=${nodes}, time=${runtime}"
# make a name variable which excludes the dir and the .py extension
name="${python_file/submitted_jobs\//}"
name="${name/%.py/}"
# set up SBATCH
# -------------
# runtime
ntasks=$(($nodes * 32))
sed -i '/#SBATCH --nodes=/c\#SBATCH --nodes='"${nodes}" 'sbatch_peta'
sed -i '/#SBATCH --ntasks=/c\#SBATCH --ntasks='"${ntasks}" 'sbatch_peta'
sed -i '/#SBATCH --time=/c\#SBATCH --time='"${runtime}" 'sbatch_peta'
# file names
sed -i '/#SBATCH -J /c\#SBATCH -J '"${name}" 'sbatch_peta'
sed -i '/#SBATCH -o slurm\//c\#SBATCH -o slurm\/'"${name}"'-%A.out' 'sbatch_peta'
# #SBATCH -o slurm/python-%A.out
sed -i '/application=/c\application='"${python_file}" sbatch_peta
sbatch sbatch_peta
## call arguments verbatim
#$@
