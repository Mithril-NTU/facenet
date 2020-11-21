set -x
python src/train_tripletloss.py \
--logs_base_dir ./yaxu_logs/facenet_${3}/ \
--models_base_dir ./yaxu_models/facenet_${3}/ \
--data_dir $1 \
--image_size 160 \
--model_def models.inception_resnet_v1 \
--lfw_dir $2 \
--optimizer RMSPROP \
--learning_rate 0.01 \
--weight_decay 1e-4 \
--gpu_memory_fraction 0.8 \
--epoch_size 100 \
--max_nrof_epochs 500
