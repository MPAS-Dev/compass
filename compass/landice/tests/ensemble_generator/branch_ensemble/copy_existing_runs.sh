#!/bin/bash
# This script can be used to copy over all of the contents of an existing ensemble
# to a fresh generation of the ensemble directory structure.
# This is useful if your compass environment gets corrupted or similar
# situation requiring you to start over with creating the environment.

SRC_PATH=/pscratch/sd/h/hoffman2/AMERY_corrected_forcing_6param_ensemble_2023-03-18_branch_runs_yr2050_2023-07-28_noGroundedCalving_noCalvingError_calvingMetrics_200runsFiltered/landice/ensemble_generator/branch_ensemble
DEST_PATH=/pscratch/sd/h/hoffman2/AMERY_corrected_forcing_6param_ensemble_2023-03-18_branch_runs_yr2050_2023-08-31/landice/ensemble_generator/branch_ensemble

for dir in ${SRC_PATH}/run* ; do
    echo $dir
    run=`basename $dir`
    cp ${dir}/log.landice* ${DEST_PATH}/${run}/
    cp ${dir}/restart_timestamp ${DEST_PATH}/${run}/
    cp ${dir}/rst*nc ${DEST_PATH}/${run}/
    cp ${dir}/uq_run*.o* ${DEST_PATH}/${run}/
    cp -R ${dir}/output ${DEST_PATH}/${run}/
done
