#!/bin/bash
force=
directory="$(dirname "$0")"/jobs
light=
while :; do
    case $1 in
	-f )
	    force=True ;;
	-d )
	    shift
	    directory=$1
	    echo $directory
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
	rm log/train.log.$job 2> /dev/null && echo Log file deleted || echo No log file found
	rm out/train-$job.* 2> /dev/null && echo Output files deleted || echo No output file found
	if [ "$light" ]
	then
	    touch "$dir"/deleted
	else
	    rm -r "$dir"
	fi
    else
	echo "$job" 'not found'
    fi

done
