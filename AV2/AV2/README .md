# AV2 – Classificação de imagens com extração de características (PlantVillage)

Equipe: Kauê Cruz e Abimael · Disciplina: Ciência de Dados  
Dataset: https://github.com/spMohanty/PlantVillage-Dataset (pasta `raw/color`)

Somente técnicas tradicionais de PDI e aprendizado de máquina (OpenCV, scikit-image, scikit-learn). **Sem CNN, deep learning ou modelos pré-treinados.**

## Como executar (PyCharm)
1. Abra a pasta do projeto, crie um interpretador (venv) e instale: `pip install -r requirements.txt`.
2. Requisitos: Python 3.9+, `git` no PATH (para o download parcial do dataset, ~1 GB) e ~3 GB livres.
3. Execute `run_all.py` (ou cada `stepXX_*.py`). Os resultados vão para `outputs/`.
   * `python run_all.py --from 4` recomeça de um passo; `--force` ignora caches.
   * Variáveis opcionais: `PV_REPO_DIR` (se já clonou o dataset) e `AV2_OUT_DIR`.
4. Todos os passos usam cache e podem ser interrompidos e retomados.

## Passos
| Script | Tópico | Principais saídas |
|---|---|---|
| `step01_dataset_analysis.py` | 1 – análise e divisão sem vazamento | `outputs/dados/inventario_imagens.csv`, `split_treino_teste.csv`, gráficos `01_*`, tabelas `01_*` |
| `step02_preprocessing.py` | 2 – pré-processamento | `02_*`, `decisao_preprocessamento.json` |
| `step03_segmentation.py` | 3 – segmentação | `03_*`, `decisao_segmentacao.json` |
| `step04_feature_extraction.py` | 4 – características e DataFrames | `features_imagem_completa.csv`, `features_imagem_segmentada.csv` |
| `step05_train_evaluate.py` | 5 – classificadores e avaliação | `05_*`, matrizes de confusão, `outputs/modelos/` |
| `step06_analysis_report.py` | 6 – análise e relatório | `outputs/relatorios/RELATORIO_AV2.md` |

## Desempenho
* **Modo rápido** (`python run_all.py --rapido` ou `python step05_train_evaluate.py --rapido`): Random Forest com 100 árvores e
  `max_features ∈ {sqrt, 0,1}`, busca em subamostra de 4.000 imagens. O Random Forest é ~80% do tempo do Passo 5; no modo rápido a
  tarefa mais pesada cai de ~22 para ~3,5 min (1 núcleo) com perda de F1 macro de ~0,007 (medido em uma tarefa). Os resultados do modo rápido
  ficam em cache separado (`outputs/cache/resultados_rapido`); o relatório do Passo 6 informa qual modo foi usado.
* `AV2_TASK_JOBS=N` define quantas tarefas rodam em paralelo no Passo 5 (padrão 4; cada uma usa ~0,5-1 GB de RAM).
* `outputs/tabelas/05_tempos_por_tarefa.csv` mostra o tempo de cada tarefa.
Extração paralela (joblib) em lotes com retomada; LBP calculado uma vez por imagem; busca de hiperparâmetros em subamostra do treino; tarefas de classificação executadas em paralelo.
Em 1 núcleo: extração ≈ 12 ms/imagem; com N núcleos o tempo cai aproximadamente 1/N.

## Organização no GitHub (pasta `AV2` da sua branch)
```
AV2/
  README.md  requirements.txt  .gitignore
  config.py utils.py features.py step01..step06 run_all.py
  outputs/graficos/  outputs/tabelas/  outputs/relatorios/RELATORIO_AV2.md
  outputs/dados/ (split, decisões JSON; os CSVs de features têm ~200 MB: use Git LFS ou publique link)
```
O `.gitignore` já exclui o dataset, `outputs/cache/` e `outputs/modelos/`.
Comandos: `git checkout -b av2-kaue-abimael`, `git add AV2`, `git commit -m "AV2 PlantVillage"`, `git push -u origin av2-kaue-abimael`.

## Metodologia em uma frase
Divisão treino/teste **agrupada por folha** (leaf-map oficial + blocos validados), pré-processamento e segmentação decididos **apenas no treino**, hiperparâmetros por validação cruzada agrupada **no treino**, padronização dentro de `Pipeline`, teste reservado para a avaliação final.
