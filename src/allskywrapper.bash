#!/bin/bash
# This will run <ntotal> points using <nthreads> simultaneous processes
# The main PySSED folder should be at location <FOLDER>
# Output will be saved to <OUTFOLDER>
ntotal=100
nthreads=4
FOLDER="original"
OUTFOLDER="output"
startdec=90
enddec=-90

conda activate explore
#echo ">>>start"
#echo $PATH

# Generate RA/Dec combinations
boxsize=0.2
rm regions.dat
touch regions.dat

box2=`echo ${boxsize} | awk '{print $1*2}'`
echo "cone 0 90 ${box2}" >> regions.dat
echo ${boxsize} | awk '{for (dec=90-$1*3;dec>=-90+$1*2;dec-=$1) {u=$1/cos((dec+($1/2))/180*pi); d=360/int(360/u+0.5); for (ra=0;ra<360-d/2;ra+=d) print "box",ra,dec,ra+d,dec+$1}}' pi=3.141526535 >> regions.dat
echo "cone 0 -90 ${box2}" >> regions.dat
wc -l regions.dat

# Clear any previous runs
rm inprogress.*
rm complete.*
rm -r thread.*

awk '$2>=s && $2<=e' s="${startdec}" e="${enddec}" regions.dat > locations.dat

# Make output directory if it doesn't exist
if [ ! -d ${OUTFOLDER} ]
then
	mkdir ${OUTFOLDER}
fi

# Main loop
n=0
ntotal=`wc -l locations.dat | awk '{print $1}'`
while [ $n -lt $ntotal ]
do
	#echo $n
	for (( i=1; i<=${nthreads}; i++ )); do
		if [ -f complete.${i} ] || [ ! -f inprogress.${i} ]
		then
		
			# Copy and clear previous data if any
			if [ -d thread.${i}/ ]
			then
				justdone=`head -1 inprogress.${i}`
				nstars=`wc -l thread.${i}/output.dat | awk '{print $1-10}'`
				totalstars=`expr $totalstars + $nstars`
				echo "Thread ${i}: finished run ${justdone} [${nstars} stars, total output ${totalstars}]"
				cp thread.${i}/output.dat ${OUTFOLDER}/output_${n}.dat
				rm inprogress.${i}
				rm -r thread.${i}/
			fi

			# Start new data
			n=`expr $n + 1`
			cmd=`head -${n} locations.dat | tail -1`
			echo "Thread ${i}: beginning run ${n}"
			echo "$n" > inprogress.${i}
			cp -r ${FOLDER}/ thread.${i}/
			cd thread.${i}/src/
			{ python3 pyssed.py ${cmd} simple setup.default ; sleep 0.1 ; touch ../../complete.${i}; } &
			cd ../..

		fi
	done
	sleep 1 # test every <x> seconds
done

echo "Waiting for final processes to finish..."
wait
echo "All processes done. Tidying up."
rm inprogress.*
rm complete.*
rm -r thread.*
