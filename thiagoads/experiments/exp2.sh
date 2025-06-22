#!/bin/bash

# +--------------------------------------------------------------------+
# | Parâmetro          |  Descrição                                    |
# +--------------------------------------------------------------------+
# | DAD_EXP            |  Identificador do Experimento                 |
# | DAD_MODEL_PATH     |  Caminho de salvamento da experiemento/modelo |
# | DAD_TM             |  Target Model                                 |
# | DAD_AM             |  Arbitray Model                               |
# | DAD_TD             |  Target Dataset                               |
# | DAD_AD             |  Arbitrary Dataset                            |
# | DAD_ATTACK         |  Método de Ataque                             |
# | DAD_TD_IMG_SIZE    |  Tamanho da imagem do target dataset          |
# | DAD_AD_IMG_SIZE    |  Tamanho da imagem do arbitrary dataset       |
# +--------------------------------------------------------------------+

export DAD_EXP="experiment_2"               
export DAD_MODEL_PATH="resnet18"               
export DAD_TM="resnet18"       
export DAD_AM="resnet18"       
export DAD_TD="cifar10"                  
export DAD_AD="fmnist"                   
export DAD_ATTACK="fgsm"
export DAD_TD_IMG_SIZE=32
export DAD_AD_IMG_SIZE=28


$(dirname "$0")/run.sh