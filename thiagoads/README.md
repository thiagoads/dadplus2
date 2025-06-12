
# Visão Geral


## Configurando o ambiente

```
# clonando o projeto do gthub
git clone https://github.com/thiagoads/dadplus2.git data-free-adversarial-defense

# entrando na pasta do projeto
cd data-free-adversarial-defense

# acessando as customizações
cd thiagoads/environment

# criando o ambiente com Python 3.10
conda env create -f environment.yml

# alternativamente podemos criar assim
# conda create -n dad++ python=3.10.0

# ativando ambiente conda
conda activate dad++

# instalando dependências (estáveis)
pip install -r requirements_stable.txt

# voltando para pasta raiz
cd -
```

## Executando experimentos
(Verificar se scripts tem permissão de execução)

### Baseline
``` 
./thiagoads/experiments/base.sh
``` 

### Experimento 1
``` 
./thiagoads/experiments/exp1.sh
``` 

### Experimento 2
``` 
./thiagoads/experiments/exp2.sh
``` 

## Monitoramento com Wandb
https://wandb.ai/thiagoads/dad%2B%2B
