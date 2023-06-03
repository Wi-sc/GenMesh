date_str=`date '+%Y-%m-%d'`
save_dir=./log/${date_str}_0shot_viewer
mkdir ${save_dir}
python -m train_viewer_center_nmr.py --batch_size=12 --gpu=0 --deform_num=3 --silhouette_weight=0.01 --save_dir=${save_dir} 2>&1|tee ${save_dir}/log_train.txt