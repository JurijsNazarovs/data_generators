#!/bin/bash

#matlab="/Applications/MATLAB_R2020b.app/bin/matlab -nosplash  -nodesktop --nojvm"
scriptdir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
#echo $scriptdir; exit 0
filepath=${1:-"..//tadpole_prediction/"}
ts=${2:-"2 4 6"}
method=${3:-'real'} #real, best, mean, extrapolation
for exp in $filepath/*tadpole*; do
  expname=$(echo $exp | cut -d _ -f 3-4)
  echo $expname
  for f in "$exp"/*.pickle; do
    fmat=${f%.*}.mat
    echo $fmat
    python3 "$scriptdir"/make_mat.py $f $fmat
    if [[ $? -ne 0 ]]; then
      echo "Failed: Python script to transfer from npy to mat."
      echo "on experiment $exp and file $f."
      exit 1
    fi

  done

  
  for f in "$exp"/*.mat; do
    for t in $ts; do
      for method in $method; do
        #real best mean; do
        plotspath="plots/$expname/"
        matlab -nosplash  -nodesktop -sd "$scriptdir" -r "plot_brain(\"$f\", $t, \"$plotspath\", \"$method\");quit"
        if [[  $? -ne 0 ]]; then
          exit 1
        fi
      done
    done
  done
 
done

