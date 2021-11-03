#!/bin/bash

cd "$(dirname "$0")"

for job in "$@"
do
    echo working on "$job"
    dir=`find . -type d -name $job`

    if [ "$dir" ]
    then
	while true; do
	    read -p "Do you wish to erase folder $1? [yn]" yn
	    case $yn in
		[Yy]* ) break;;
		[Nn]* ) exit;;
		* ) echo "Please answer yes or no. ";;
	    esac
	done
	echo "$dir" 'will be deleted'
	rm log/train.log.$1 2> /dev/null && echo Log file deleted || echo No log file found
	rm out/job-$1.* 2> /dev/null && echo Output files deleted || echo No output file found
	rm -r "$dir"
    else
	echo "$job" 'not found'
    fi

done
