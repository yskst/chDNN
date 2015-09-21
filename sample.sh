#!/bin/bash

# This is a sample script of chDNN.
# This script train Deep Neural Network after trianing RBMs.

# Configuration----------------------------------------------------------

# The number of each layers' node.
# The number of first layer's node specifies input dimension.
# If the purpose of DNN is classification, the number of last layer's node
# must be correspond to the number of class.
nnode=(26 1000 500 20 3)
dnntype=c       # The purpose of DNN. c:classification / f:regression.
actfunc=sigmoid # Activate function.(e.g. sigmoid, tanh, ReLU)

train=train.dat    # Training data.
train_fromat=text  # The format of training data.(e.g. f4le, text, npy)
target=target.dat  # Supervise data. 
target_format=text # Th format of target data.(e.g. f4le, text, npy)

outdir=. # The directory to store intermediate and trained datas.


# Configuration of RBM training.
rbm_mb=64     # mini-batch size
rbm_epoch=100 # number of iteration
rbm_lr=1e-3   # learning rate
rbm_mm=1e-4   # momentum coefficient.
rbm_re=1e-5   # L2-reguralization coefficient.


# Configuration of backpropagation.
bp_mb=64
bp_epoch=100
bp_lr=1e-3
bp_mm=1e-4
bp_re=1e-5


#-----------------------------------------------------------------------
scpdir=`dirname $0`
nlayer=${#nnode[*]}
mkdir -p $outdir/rbm $outdir/mlp $outdir/log

rbms=()

echo "Training RBM to initialize DNN."
data=$train
format=$train_fromat
rbmtype=gb   # The input data is gaussian distribution.
e=`expr $nlayer - 2`
for i in `seq 0 $e`;do
  j=`expr $i + 1`
  if [ $i -eq $e ]; then
    rbm_epoch=0
    if [ dnntype = "c" ];then
      $actfunc=softmax
    fi
  fi

  echo "Train ${i}-th RBM..."
  $scpdir/rbmtrain.py --of $outdir/rbm/L$i.npz   --df $format  \
                      --mb $rbm_mb  -e $rbm_epoch              \
                      --lr $rbm_lr  --mm $rbm_mm --re $rbm_re  \
                      --rt $rbmtype --af $actfunc              \
                      ${nnode[$i]} ${nnode[$j]} $data         |\
    tee $outdir/log/L$i.log

  echo "Forward propagation to train next layer's RBM..."
  $scpdir/fprop.py --nn $outdir/rbm/L$i.npz --of $outdir/rbm/L$i.dat \
                   --ot f4ne --df $format 
  rbmtype=bb
  data=$outdir/rbm/L$i.dat
  format=f4ne
  rbms+=("$outdir/rbm/L$i.npz")
done

echo "Concatenate RBMs...
$scpdir/rbms2mlp.py --of $oudir/mlp/nn_init.npz ${rbms[@]}

echo "Back propagation..."
$scpdir/backprop.py --mlp $outdir/mlp/nn_init.npz --of $outdir/mlp/nn.npz \
                    --df $train_fromat --tf $target_format --tt $dnntype  \
                    --mb $bp_mb --e $bp_epoch                             \
                    --lr $bp_lr --mm $bp_mm --re $bp_re                   \
                    $train $target                                       |\
  tee $outdir/log/bp.log
echo "Save trained DNN parameter into $outdir/mlp/mm.npz"

