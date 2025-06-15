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
export DAD_MODEL_PATH="mobilenet"               
export DAD_TM="mobilenet_v3_small"       
export DAD_AM="mobilenet_v3_small"       
export DAD_TD="cifar10"                  
export DAD_AD="rival10"                   
export DAD_ATTACK="fgsm"
export DAD_TD_IMG_SIZE=32
export DAD_AD_IMG_SIZE=224


$(dirname "$0")/run.sh