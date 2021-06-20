#!/bin/bash
sync=
remote=lab-ia
compile=
feat=
while :; do
    case $1 in
        --sync ) 
	    sync=true
	    if [ $2 ] && [[ $2 != -* ]]; then
		shift
		remote=$1
		shift
	    fi
            # break 
            ;;
	--tex )
	    # shift
	    compile=tex
	    # break
	    ;;
	-f )
	    shift
	    feat=$1
	    shift
	    ;;
	* )
	    break
	    ;;
    esac
    shift
done

cd ~/python/dnn/joint-vae

[ $compile ] && echo TeX file will be compiled || echo No compilation of TeX file 
if [ $sync ]; then
    echo Jobs will be synced from $remote
    if [ $remote = lab-ia ]; then
	. jobs/rsync-jobs-lab-ia
    elif [ $remote = lss ]; then
	. jobs/rsync-jobs-lss
    fi
fi

source ~/.virtualenvs/cuda/bin/activate

test='python test.py -e --tnr --sort dict-var beta-sigma'

###
# LATEX CODE
# \resset\restype\resopts\resjobs\resfeatures\resdepth\resdone\resdataaug\ressigmatrain



#### SVHN
#
# \def\resset{--set-svhn}
# \def\restype{--type-cvae-or-vae}
# \def\resopts{--coder-dict-not-learned}
# \def\resjobs{}
# \def\resfeatures{--features-conv-p1-64-128-256}
# \def\resdepth{--depth-1}
# \def\resdone{--done-20..inf}
# \def\resdataaug{--data-augmentation-not-flip--data-augmentation-crop}
# \def\ressigmatrain{}


s='--dataset svhn'
t='--type cvae vae'
o='--coder-dict not learned --batch-norm both'
j=''
# f='--features conv-p1-64-128-256'
if [ $feat ]; then
    f_=("--features $feat")
else
    f_=('--features conv-p1-64-128-256' '--features vgg16')
fi
d_=('--depth 1' '--depth 5')
e='--epochs 80..'
da='--data-augmentation not flip --data-augmentation crop'
st_=('--sigma-train constant' '--sigma-train learned' '--sigma-train not rmse')


for f in "${f_[@]}"; do
    for st in "${st_[@]}"; do
	for d in "${d_[@]}"; do
	    echo '######'
	    echo $test $s $t $o $j $f $d $e $da $st
	    $test $s $t $o $j $f $d $e $da $st
	done
    done
done

### CIFAR

## $test --dataset cifar10 --depth 1 --type cvae vae --epochs 20.. --sigma-train constant --job 109500..


### ANNEXE

# $test --dataset fashion --type cvae --epochs 20.. --best-acc 0.75..
## $test --dataset fashion --type cvae --epochs 10.. --job 109500..

## $test --dataset svhn --type cvae --epochs 20.. --best-acc 0.5..
# $test --dataset svhn --depth 1 --type cvae --epochs 20.. --sigma-train constant --job 109500..
# $test --dataset svhn --depth 1 --type vae --epochs 20.. --sigma-train constant --job 109500..
# $test --dataset svhn --depth 1 --type cvae --epochs 20.. --sigma-train rmse --job 109500..
# $test --dataset svhn --depth 1 --type vae --epochs 20.. --sigma-train rmse --job 109500..

## $test --dataset cifar10 --type cvae --epochs 20.. --best-acc 0.5..

if [ $compile ]; then
    cd ~/doctorat/notes/jvae/
    latexmk jvae.tex -g
fi
