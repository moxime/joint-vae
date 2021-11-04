#!/bin/bash
force=
while :; do
    case $1 in
	-f )
	    force=True ;;
	* )
	    break ;;
    esac
    shift
done
echo Force delete: $force_delete
echo Working on "$(dirname "$0")"
cd "$(dirname "$0")"

for job in "$@"
do
    echo Searching "$job"
    dir=$(find . -type d -regex ".*/0*$job")
    if [ "$dir" ]
    then
	while [ -z $force ]; do
	    read -p "Do you wish to erase folder $dir? [yn]" yn
	    case $yn in
		[Yy]* ) break;;
		[Nn]* ) exit;;
		* ) echo "Please answer yes or no. ";;
	    esac
	done
	scancel "$job" 2> /dev/null
	echo "$dir" 'will be deleted'
	rm log/train.log.$job 2> /dev/null && echo Log file deleted || echo No log file found
	rm out/job-$job.* 2> /dev/null && echo Output files deleted || echo No output file found
	rm -r "$dir"
    else
	echo "$job" 'not found'
    fi

done
