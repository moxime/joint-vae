#!/bin/bash
remote=lab-ia
push=
target=$(dirname $0)
opt=( --exclude '*.pth' -u )
while :; do
    case $1 in
	--push )
	    push=True
	    ;;
	--flash )
	    opt=( --exclude '*.pth' -u )
	    ;;	    
	--light )
	    opt=( --include 'record-*.pth' --exclude '*.pth' -u )
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
    remote=$1
fi
shift

if [ "$1" = "lab-ia" ]
then
    source="~/joint-vae/jobs"
else
    source="~/python/dnn/joint-vae/jobs"
fi

if [ -z $push ]
then
   from=$remote:$source/
   to=$target/
else
   to=$remote:$source/
   from=$target/

fi
   
echo rsync  "${opt[@]}" $@ $from $to

exit 0
SECONDS=0
# rsync -vaP lab-ia:/mnt/beegfs/home/ossonce/joint-vae/jobs/ jobs/
# ssh lab-ia '. mirror-jobs.sh'

rsync -a "${opt[@]}" --exclude "log/*" --exclude "out/*" $@ $from $to
duration=$SECONDS
echo "Files retrieved in $(($duration / 60))m$(($duration % 60))s"
