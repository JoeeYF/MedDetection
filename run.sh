#python train.py --config ./configs/cfg_seg_luna16_nodule_UConvLSTM.py
#sleep 100
#python train.py --config ./configs/cfg_seg_APCTP_frangi.py

#for i in {0..2}
#do
#  echo $i
#  python ./train.py
#  python ./infer.py
#  python ./eval_drive.py
#done

config=$1
train=$2
infer=$3
eval=$4

FOLD=9

config_file="configs/${config}.py"
echo "config = ${config_file}"

if [ "${train}" = '1' ]; then
  echo "train"

  python ./train.py  --config "${config_file}"
fi

if [ "${infer}" = '1' ]; then
  echo "infer"

  for i in {1..15}
  do
    if test -d "work_dirs/${config}/infer_results_${FOLD}_${i}0ep" ; then
      continue
    fi

    checkpoint_file="work_dirs/${config}/epoch_${i}0.pth"
    echo "checkpoint = ${checkpoint_file}"
    python ./infer.py --config "${config_file}" --ckpt "${checkpoint_file}"

    mv "work_dirs/${config}/infer_results" "work_dirs/${config}/infer_results_${FOLD}_${i}0ep"

  done
fi

if [ "${eval}" = '1' ]; then
 echo "eval"

 python ./evaluationScript/frocwrtdetpepchluna16_medtk_Luna2016.py --config "${config}" --fold ${FOLD}
fi