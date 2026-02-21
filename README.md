# Diagnóstico Probabilístico de Pneumonia em Raio-X Torácico com Aplicação em Visão Computacional e Explicabilidade Grad-CAM

## Sobre o Projeto

Este projeto implementa um sistema de visão computacional para **diagnóstico probabilístico de pneumonia** em imagens de raio-x de tórax, utilizando técnicas de transfer learning com a arquitetura **DenseNet121** pré-treinada no ImageNet.

* Link do relatório técnico: https://drive.google.com/file/d/1gcG6V2BiuZQbjLyPOlXUqSi1KGi5-9k2/view?usp=sharing

### Características

- **Saída probabilística (0-1)** ao invés de classificação binária
- **Explicabilidade via Grad-CAM** para visualização de regiões de atenção
- **Pesos clínicos** priorizando sensibilidade (recall) para minimizar falsos negativos

---

### Arquitetura do Modelo

- **Base:** DenseNet121 pré-treinada
- **Treinamento em duas fases:**
  - Fase 1: Feature extraction (15 epochs)
  - Fase 2: Fine-tuning (25 epochs)
- **Saída:** Sigmoid para probabilidades contínuas

### Métricas

- AUC-ROC
- Recall (Sensibilidade)
- Matriz de Confusão

---

## 📂 Estrutura do Projeto
```
PROJETO-LIGIA---CHEST-X-RAY/
│
├── Notebooks/
│   ├── Análise exploratória/
|   |   ├── EDA_X_ray.ipynb            
│   ├── Modelagem/
│   |   ├── Modelagem descartada/              
|   |   |   ├── Modelagem_Cross_validation_X_ray.ipynb       
│   |   ├── Modelagem Final/              
|   |   |   ├── Modelagem_final_X_ray.ipynb       
│   └── Tratamento de dados/
|       ├── Tratamento_dados_X_ray.ipynb 
│
├── Modelo
│   ├── modelo_final .h5                  
│
├── Resultados/
│   └── submission.csv                  # Arquivo de submissão Kaggle
│
├── requirements.txt                    # Dependências pip
├── README.md                           # Este arquivo
```
## Dataset e modelo
Como os arquivos excedem o tamanho máximo permitido no GitHub, o dataset e o modelo pronto estão salvos no Google Drive nos seguintes links:

* Modelo final (h5): https://drive.google.com/file/d/14SxjeH-ahaupFexF5QBfDL8p9KR0vZ72/view?usp=sharing
* Dataset de treino (zip): https://drive.google.com/file/d/1IBC3mk83DnHkZ4Xn3Kq9H18kM3_WITx2/view?usp=sharing
* Dataset de treino tratado (zip): https://drive.google.com/file/d/1AGEiWDB7BaZA0qrTSsZOkMqM7fd9ru4e/view?usp=sharing
* Dataset de treino tratado (csv): https://drive.google.com/file/d/14IXoOOZ2Wcxco2Asd3UqZEG01zSbzfud/view?usp=sharing
* Dataset de teste (zip): https://drive.google.com/file/d/1ub-8oqdHQl7PI6oZxdaz1coUIYJfeRII/view?usp=sharing
* Dataset de teste (csv): https://drive.google.com/file/d/1cxBQ3_JkpxuLNwGqyHjpzKnC1CwPSlz_/view?usp=sharing

---

## Uso

## 1. Baixando o repositório
'''
Opções:

* 1. Clonar via git:

```bash
git clone https://github.com/MClarapg/Projeto-Ligia---Chest-X-ray.git
```

* 2. Ou baixar ZIP:

  * Acesse a página do repositório e clique em **Code → Download ZIP**.
  * Extraia o ZIP em uma pasta local.
 
Após baixar, você terá a estrutura com os notebooks, datasets, modelo e arquivos auxiliares.

---

## 2. Abrir notebooks no Google Colab

Opções para abrir um notebook do seu computador no Colab:

* No Colab: **File → Open notebook → Upload** → selecione o arquivo `.ipynb` baixado.

---

## Observação sobre o dataset

Não é necessário fazer o upload do dataset nos notebooks, há células iniciais que carregam o dataset diretamente de pastas do Google Drive. O dataset disponibilizado no GitHub é para controle do material utilizado.

---

## 3. Ordem de execução recomendada dos notebooks

### 4.1. `EDA_X_Ray.ipynb`

* Objetivo: Análise Exploratória e visualização de amostras do dataset.

### 4.2. `Tratamento_de_dados_X_ray.ipynb`

* Objetivo: Limpeza e tratamento dos dados. 
* Neste notebook, será baixado um zip contendo os dados tratados, aceitar o download é opcional, pois o dataset tratado já está salvo no GitHub e no drive.

### 4.3. `Modelagem_Final_X_ray.ipynb`

* Objetivo: treinamento, validação e avaliação do modelo.
* Neste dataset, um arquivo csv será baixado no final das execuções, contendo a saída de submissão desejada para o Kaggle. Não é necessária a tomada de qualquer ação adicional além da permissão de download.

---