#!/bin/bash
remote=lab-ia
jobdir=jobs
push=
opt=( --exclude '*.pth' -uvP --exclude '*.out' )
while :; do
    case $1 in
	-j )
	    shift
	    jobdir="$1"
	    ;;
	--push )
	    push=True
	    ;;
	--flash )
	    opt=( --exclude '*.pth' -uvP )
	    ;;	    
	--light )
	    opt=( --include 'record-*.pth' --exclude '*.pth' -uvP )
	    ;;
	--full )
	    opt=( --exclude '*/optimizer.pth' -uvP )
	    ;;
	--fullest )
	    opt=( -uvP )
	    ;;
	-x )
	    donotdelete=true
	    ;;
	* )
	    break
	    ;;
    esac
    shift
done
if [ $1 ]
then
    remote=$1
fi
shift

target=$(dirname $0)/../$jobdir

source="~/joint-vae/$jobdir"

if [ -z $push ]
then
   from=$remote:$source/
   to=$target/
else
   to=$remote:$source/
   from=$target/
fi

echo rsync  "${opt[@]}" $@ $from $to

# exit 0
SECONDS=0
# rsync -vaP lab-ia:/mnt/beegfs/home/ossonce/joint-vae/jobs/ jobs/
# ssh lab-ia '. mirror-jobs.sh'

rsync -a "${opt[@]}" --exclude "log/*" --exclude "out/*" $@ $from $to | tee /tmp/downloaded-$remote
grep architecture.json /tmp/downloaded-$remote
duration=$SECONDS
echo "Files retrieved in $(($duration / 60))m$(($duration % 60))s"

if [ -z $push ] && [ -z $donotdelete ];
then
    echo deteling models-\*.json files
    rm "$target"/models-*.json
fi
