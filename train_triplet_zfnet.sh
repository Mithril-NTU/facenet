set -x
python src/train_tripletloss.py \
--logs_base_dir ./yaxu_logs/facenet_${5}/ \
--models_base_dir ./yaxu_models/facenet_${5}/ \
--data_dir $1 \
--people_per_batch 45 \
--images_per_person 40 \
--image_size 160 \
--model_def models.zfnet \
--lfw_dir $2 \
--optimizer RMSPROP \
--learning_rate 0.01 \
--weight_decay 1e-4 \
--gpu_memory_fraction 0.9 \
--epoch_size 240 \
--max_nrof_epochs 500 \
--lfw_pairs $3 \
--full_label_matrix $4
#--evaluate_by_auc
