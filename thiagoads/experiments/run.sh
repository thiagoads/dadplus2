#!/bin/bash

echo "Iniciando experimento com os seguintes parâmetros:"
export | grep DAD_

echo "Criando diretórios necessários..."
mkdir -p ./checkpoints/${DAD_EXP}/${DAD_MODEL_PATH}/${DAD_TD}
mkdir -p ./checkpoints/${DAD_EXP}/${DAD_MODEL_PATH}/${DAD_AD}
mkdir -p ./logs


# stage1: treinamento do target model
python train_model.py                                                \
--name "${DAD_EXP}_stage_1"                                          \
--dataset ${DAD_TD}                                                  \
--batch_size 32                                                      \
--lr 0.01                                                            \
--image_size ${DAD_TD_IMG_SIZE}                                      \
--epochs 100                                                         \
--model_name ${DAD_TM}                                               \
--save_path ./checkpoints/${DAD_EXP}/${DAD_MODEL_PATH}/${DAD_TD}/target_model.pt \
--wandb                                                              \
#--thiagoads_subset_param 0.01                                        \



# stage2: treinamento do arbitrary model
python train_model.py                                                \
--name "${DAD_EXP}_stage_2"                                          \
--dataset ${DAD_AD}                                                  \
--batch_size 64                                                      \
--lr 0.01                                                            \
--image_size ${DAD_AD_IMG_SIZE}                                      \
--epochs 50                                                          \
--model_name ${DAD_AM}                                               \
--save_path ./checkpoints/${DAD_EXP}/${DAD_MODEL_PATH}/${DAD_AD}/arbitrary_model.pt \
--wandb                                                              \
#--thiagoads_subset_param 0.01                                        \



# stage3: treinamento do source detector para reconhecer ataques
python train_arbitary_detector.py                                     \
--name "${DAD_EXP}_stage_3"                                           \
--dataroot clean_data/${DAD_AD}                                       \
--dataset ${DAD_AD}                                                   \
--batch_size 128                                                      \
--model_name ${DAD_AM}                                                \
--model_path ./checkpoints/${DAD_EXP}/${DAD_MODEL_PATH}/${DAD_AD}/arbitrary_model.pt \
--attack ${DAD_ATTACK}                                                \
--gpu 0                                                               \
--method vanila                                                       \
--epochs 10                                                           \
--seed 0                                                              \
--use_wandb                                                           \
#--thiagoads_subset_param 0.01                                         \



# stage4: treinamento do target detector com UDA e avaliação diante de ataques
python combined.py                                                   \
--name "${DAD_EXP}_stage _4"                                         \
--dataset ${DAD_TD}                                                  \
--batch_size 64                                                      \
--model_name ${DAD_TM}                                               \
--model_path ./checkpoints/${DAD_EXP}/${DAD_MODEL_PATH}/${DAD_TD}/target_model.pt \
--detector_path ./checkpoints/${DAD_EXP}/${DAD_MODEL_PATH}/${DAD_AD}/${DAD_AD}_${DAD_ATTACK}_seed_0_source_detector.pt \
--attacks ${DAD_ATTACK}                                              \
--method vanila                                                      \
--gpu 0                                                              \
--droprate 0.005                                                     \
--seed 0                                                             \
--lr 0.005                                                           \
--epochs 10                                                          \
--s_model ${DAD_AM}                                                  \
--s_dataset ${DAD_AD}                                                \
--ent_par 0.8                                                        \
--cls_par 0.3                                                        \
--correction_batch_size 256                                          \
--r_range 16                                                         \
--soft_detection_r 32                                                \
--log_path ./logs/logs_balanced.txt                                  \
--pop 10                                                             \
--retrain_detector                                                   \
--recreate_adv_data                                                  \
--use_wandb                                                          \
#--thiagoads_subset_param 0.01                                        \

echo "Experimento concluído com sucesso!"
