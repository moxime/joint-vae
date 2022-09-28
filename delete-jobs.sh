#!/bin/bash
force=
script_abs_path=$(realpath "$0")
root_directory=$(dirname "$script_abs_path")

directory="$(dirname "$0")"/jobs
light=
out_dir="$root_directory"/jobs/out
log_dir="$root_directory"/jobs/out/log

while :; do
    case $1 in
	-f )
	    force=True ;;
	-d )
	    shift
	    directory=$1
	    ;;
	-l )
	    light=True
	    ;;
	* )
	    break ;;
    esac
    shift
done
echo Force delete: $force
echo Light delete: $light
echo Working on $directory
cd "$directory"

for job in "$@"
do
    echo Searching "$job"
    dir=$(find . -type d -regex ".*/0*$job" -not -regex ".*/samples/0*$job")
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
	if [ "$light" ]
	then
	    touch "$dir"/deleted
	else
	    rm -r "$dir"
	fi
    else
	echo "$job" 'not found'
    fi

    rm "$log_dir"/train.log.$job 2> /dev/null && echo Log file deleted || echo No log file found
    rm "$out_dir"/train-$job.* 2> /dev/null && echo Output files deleted || echo No output file found

done
