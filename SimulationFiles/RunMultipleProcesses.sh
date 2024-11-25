#!/bin/bash
# bash to create folders, copy the input files, and sends jobs from these folder
echo 'copying files and running hashes many times'

for i in {1..5}
do 
	cp process* run_$i/
	cd ./run_$i/
	sbatch ./processToQ.sh
	cd ../
done
