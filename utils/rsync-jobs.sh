#!/bin/bash
source=lab-ia:
jobdir=jobs
target=$HOME/
opt=( --exclude '*.pth' -uvP --exclude '*.out' )
while :; do
    case $1 in
	--dry-run )
	    dry_run=True
	    ;;
	-j )
	    shift
	    jobdir="$1"
	    ;;
	--to )
	    shift
	    target="$1"
	    ;;
	--flash )
	    opt=( --exclude '*.pth' -uvP )
	    ;;	    
	--light )
	    opt=( --include 'record-*.pth' --include 'samples-*.pth' --exclude '*.pth' -uvP )
	    ;;
	--full )
	    opt=( --exclude '*/optimizer.pth' -uvP )
	    ;;
	--fullest )
	    opt=( -uvP )
	    ;;
	* )
	    break
	    ;;
    esac
    shift
done
if [ $1 ]
then
    source="$1"
fi
shift

to="${target}joint-vae/$jobdir"

from="${source}joint-vae/$jobdir/"

echo rsync  "${opt[@]}" $@ $from $to

if [ $dry_run ]
   then
       exit 0
fi

SECONDS=0

rsync -a "${opt[@]}" --exclude "log/*" --exclude "out/*" $@ $from $to #| tee /tmp/downloaded-$source-$target
# grep architecture.json /tmp/downloaded-$source-$target
duration=$SECONDS
echo "Files retrieved in $(($duration / 60))m$(($duration % 60))s"

