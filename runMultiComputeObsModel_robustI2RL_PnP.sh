#!/bin/bash

cd ~/catkin_ws
source devel/setup.bash
catkin_make

# > /home/saurabharora/Downloads/resultsApproxObsModelMetric1.csv 
# > /home/saurabharora/Downloads/resultsApproxObsModelMetric2.csv 
# > /home/saurabharora/Downloads/resultsApproxObsModelMetric3.csv 
# > /home/saurabharora/Downloads/noisyObsRobustSamplingMeirl_LBA_data.csv
# > /home/saurabharora/Downloads/noisyObsRobustSamplingMeirl_EVD_data.csv
# > /home/saurabharora/Downloads/noisyObsRobustSamplingMeirl_InfTimeAllSessions_data.csv

TRIALS=25

for ((i=1; i<=TRIALS; i++)) 
do
	./devel/bin/noisyObsRobustSamplingMeirl
done

