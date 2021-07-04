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
fold=$2
train=$3
infer=$4
eval=$5

config_file="projects/luna16_nodule_detection/${config}.py"
echo "config = ${config_file}"

if [ "${train}" = '1' ]; then
  echo "train"

  python ./train_folds.py  --config "${config_file}" --fold "${fold}"
fi

if [ "${infer}" = '1' ]; then
  echo "infer"

  for i in {1..15}
  do
    if test -d "${MEDTK}/work_dirs/${config}/${fold}/infer_results_${i}0ep" ; then
      continue
    fi

    echo "checkpoint = ${i}0"
    python ./infer_folds.py --config "${config_file}" --fold "${fold}" --ckpt "${i}0"

    mv "${MEDTK}/work_dirs/${config}/${fold}/infer_results" "${MEDTK}/work_dirs/${config}/${fold}/infer_results_${i}0ep"

  done
fi

if [ "${eval}" = '1' ]; then
  echo "eval"

  python ./projects/luna16_nodule_detection/evaluationScript/frocwrtdetpepchluna16_medtk_Luna2016.py --config "${config}" --fold "${fold}"
fi