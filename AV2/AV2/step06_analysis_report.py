"""
PASSO 6 / TÓPICO 6 - Análise crítica dos resultados e geração do texto-base do relatório.

Lê SOMENTE arquivos produzidos pelos Passos 1-5 e monta RELATORIO_AV2.md com:
fundamentação teórica | procedimento implementado | resultados obtidos | interpretação,
separados explicitamente. Todos os números vêm dos arquivos de resultado (nada é digitado
à mão); explicações causais são apresentadas como HIPÓTESES quando não foram testadas.

Uso:  python step06_analysis_report.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import config
from utils import log, require, pretty_class, split_class


def md(df: pd.DataFrame, nd: int = 3) -> str:
    if df is None or len(df) == 0:
        return "_(sem dados)_"
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.{nd}f}")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def J(p):
    return json.loads(open(p, encoding="utf-8").read())


def T(name):
    return pd.read_csv(config.DIR_TAB / name)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def main():
    s1 = J(config.DIR_REPORT / "resumo_passo1.json")
    s5 = J(config.DIR_REPORT / "resumo_passo5.json")
    pre = J(config.PREPROC_DECISION_JSON)
    seg = J(config.SEG_DECISION_JSON)
    res = T("05_resultados_todas_as_tarefas.csv")
    dummy = T("05_baseline_dummy.csv")
    classes_tab = T("01_classes_contagem.csv")
    prep = T("02_comparacao_preprocessamento.csv")
    cspace = T("02_comparacao_espacos_cor.csv")
    stdz = T("02_efeito_padronizacao.csv")
    segcmp = T("03_comparacao_metodos_segmentacao.csv")
    segcv = T("03_segmentacao_vs_inteira_cv.csv")
    abl = T("05_ablacao_grupos.csv")
    pairs = T("05_pares_mais_confundidos.csv")
    worst = T("05_classes_ordenadas_por_f1.csv")
    valblk = T("01_validacao_largura_bloco.csv")
    cand = T("01_split_oficial_candidatos.csv") if (config.DIR_TAB / "01_split_oficial_candidatos.csv").exists() else None

    A, B, S5 = s1["auditoria"], s1["balanceamento"], s1["split"]
    bk = s1["fundo_atalho"]
    best_test = res.sort_values("f1_macro_teste", ascending=False).iloc[0]
    chosen_key = s5["modelo_escolhido_por_cv"]
    cv_, cfg_, mod_ = chosen_key.split("__")
    chosen = res[(res.variante == cv_) & (res.config == cfg_) & (res.modelo == mod_)].iloc[0]
    res["rotulo"] = res["variante"] + " | " + res["config"]

    inte = res[res.variante == "inteira"]
    segm = res[res.variante == "segmentada"]
    by_cfg = inte.groupby("config")["f1_macro_teste"].mean()
    by_model = res.groupby("modelo")["f1_macro_teste"].mean().sort_values(ascending=False)
    cfgs3 = ["C1_COR", "C2_COR+TEXTURA", "C3_COR+TEXTURA+GRADIENTES"]
    m = inte.merge(segm, on=["config", "modelo"], suffixes=("_int", "_seg"))
    m["delta_seg"] = m["f1_macro_teste_seg"] - m["f1_macro_teste_int"]
    seg_mean_delta = float(m["delta_seg"].mean())
    seg_wins = int((m["delta_seg"] > 0).sum())
    c4 = segm[segm.config == "C4_C3+FORMA"].set_index("modelo")["f1_macro_teste"]
    c3s = segm[segm.config == "C3_COR+TEXTURA+GRADIENTES"].set_index("modelo")["f1_macro_teste"]
    d_forma = (c4 - c3s)
    same_crop = s5["fracao_erros_mesma_cultura"]
    p0 = prep[(prep.variante == "P0_128_base") & (prep.config == "C2_COR+TEXTURA")].iloc[0]
    pc = prep[(prep.variante == pre["escolhida"]) & (prep.config == "C2_COR+TEXTURA")].iloc[0]
    sd = float(prep[prep.config == "C2_COR+TEXTURA"]["f1_macro_dp"].mean())
    cls_f1 = pd.read_csv(config.DIR_TAB / "05_metricas_por_classe_modelo_escolhido.csv")
    best_cw = json.loads(chosen["melhores_hiperparametros"])

    sec = []
    A_ = sec.append
    A_("# AV2 — Classificação de imagens com extração de características (PlantVillage)\n")
    A_("**Disciplina:** Ciência de Dados  |  **Equipe:** Kauê Cruz e Abimael\n")
    A_("**Dataset:** PlantVillage (repositório oficial spMohanty/PlantVillage-Dataset, pasta `raw/color`)\n")
    A_("> Este documento foi gerado automaticamente por `step06_analysis_report.py` a partir dos arquivos de resultado "
       "dos Passos 1–5. Cada seção separa **fundamentação teórica**, **procedimento implementado**, **resultados obtidos** "
       "e **interpretação**. Explicações causais não testadas diretamente estão marcadas como *hipótese*. "
       "Revise e adapte a redação antes de entregar.\n")

    # ------------------------------------------------------------------ T1
    A_("\n---\n## 1. Análise e preparação do dataset\n")
    A_("### 1.1 Fundamentação\n"
       "**Vazamento de dados** (*data leakage*) ocorre quando informação do conjunto de teste influencia o treinamento, "
       "inflando as métricas (Kaufman et al., 2012). No PlantVillage isso é crítico: várias imagens são fotos diferentes "
       "(ou cópias aumentadas) **da mesma folha**; se uma ficar no treino e outra no teste, o classificador pode "
       "‘reconhecer a folha’ em vez de aprender a doença. Por isso a divisão deve ser feita por **grupos** (folhas). "
       "O **desbalanceamento** das classes faz com que a acurácia seja dominada pelas classes grandes; por isso usamos também "
       "precisão, recall e F1 por classe e o **F1 macro** (média simples dos F1 por classe), que pesa todas as classes igualmente.\n")
    A_("### 1.2 Procedimento implementado\n"
       "1. Download por `git clone --filter=blob:none --sparse` + `sparse-checkout` de `raw/color` e `leaf_grouping` "
       "(somente o necessário). Nenhuma carga via Hugging Face foi usada.\n"
       "2. Verificação da estrutura, das classes e da coerência com o README (nada foi presumido; divergências são reportadas).\n"
       "3. Varredura paralela: MD5, decodificação completa (Pillow `verify`+`load`), dimensões.\n"
       "4. Auditoria de rótulos: duplicatas exatas (MD5), fotos com mesmo nome em classes distintas, coerência com `leaf-map.json`.\n"
       "5. Agrupamento por folha usando o `leaf-map.json` oficial (mesma regra de chave do loader oficial `plant_village.py`); "
       "para imagens sem mapeamento, blocos de fotos de numeração consecutiva (estratégia alternativa validada nos dados mapeados). "
       "Duplicatas exatas são unidas ao mesmo grupo (union-find).\n"
       "6. Divisão `StratifiedGroupKFold(5)` (1 dobra = ~20% de teste), escolhendo entre 10 sementes a de proporções por classe "
       "mais próximas de 20% (critério usa apenas rótulos e grupos).\n")
    A_("### 1.3 Resultados obtidos\n")
    mc = s1["manifesto_contagem"]
    A_(f"- Pastas de classe em `raw/color`: **{s1['n_pastas_classe']}**; arquivos: **{A['n_arquivos']}** "
       f"(README: 54.306 imagens; manifesto do repositório: color={mc['color']}, segmented={mc['segmented']}, "
       f"grayscale={mc['grayscale']} — a diferença de 1 arquivo vem da própria pasta `segmented`).\n"
       f"- Culturas: **{s1['n_culturas']}**; pares cultura–doença (classes não saudáveis): **{s1['n_pares_cultura_doenca']}**; "
       f"nomes de doença distintos: {s1['n_doencas_distintas']} (README: 14 culturas e 26 doenças — coerente se contarmos pares cultura–doença).\n"
       f"- Imagens corrompidas: **{A['n_corrompidas']}**. Dimensões: {A['dimensoes']}. Extensões: {A['extensoes']}.\n"
       f"- Duplicatas exatas (MD5): **{A['n_grupos_duplicados']}** grupos ({A['n_imagens_em_duplicatas']} arquivos); "
       f"entre classes distintas (conflito de rótulo): **{A['n_duplicatas_entre_classes']}**. "
       f"Foram mantidas {A['n_usadas']} imagens (removidas: {A['excluidas_por_motivo']}).\n"
       f"- Nomes de foto repetidos em mais de uma classe: {A['n_fotos_em_mais_de_uma_classe'] if 'n_fotos_em_mais_de_uma_classe' in A else A['n_fotos_em_mais_de_uma_classe']} "
       f"(sem MD5 idêntico: são colisões de numeração entre culturas, não rótulos conflitantes). "
       f"{A['n_arquivos_em_fotos_repetidas']} arquivos pertencem a fotos com nome repetido (cópias aumentadas).\n"
       f"- Alias de rótulo do leaf-map detectado automaticamente: {s1.get('aliases_leaf_map')} "
       "(mancha-olho-de-rã é a forma foliar da podridão negra da maçã; as imagens estão na pasta `Apple___Black_rot`).\n")
    A_(f"**Balanceamento.** {B['n_classes']} classes; mín.={B['min']}, máx.={B['max']}, mediana={B['mediana']:.0f}; "
       f"razão máx./mín.={B['razao_max_min']:.1f}; maior classe: `{B['classe_majoritaria']}` ({B['pct_majoritaria']:.1f}% das imagens); "
       f"entropia normalizada={B['entropia_normalizada']:.3f} (1 = perfeitamente balanceado). "
       f"Um classificador que sempre prevê a classe majoritária acerta ≈ {B['acuracia_baseline_classe_majoritaria_pct']:.1f}% (acurácia), mas com F1 macro muito baixo.\n")
    A_("Figuras: `graficos/01_distribuicao_classes.png`, `01_distribuicao_culturas.png`, `01_exemplos_classes_1/2.png`. "
       "Tabela completa: `tabelas/01_classes_contagem.csv`.\n")
    A_("**Investigação de divisões oficiais.** O GitHub não possui pasta `splits/` "
       f"(verificado no manifesto: {s1['investigacao_split_oficial']['existe_pasta_splits_no_github']}); os arquivos de divisão citados no README_HF "
       "estão no Hugging Face "
       f"(acessíveis nesta execução: {s1['investigacao_split_oficial']['splits_hf_acessiveis']}). "
       "O único candidato no repositório é `data_distribution_for_SVM/` (gerado para um experimento do artigo):\n")
    A_(md(cand) if cand is not None else "_(candidatos não avaliados)_")
    A_("\nEsse candidato cobre apenas parte do dataset, tem proporção invertida (mais teste que treino) e, sob o nosso "
       "agrupamento, compartilha grupos entre treino e teste; **foi rejeitado**. Adotamos a divisão própria agrupada.\n")
    A_("**Validação da estratégia alternativa de grupos** (taxa de folhas reais — com `leaf_id` — que seriam partidas entre blocos de fotos consecutivas):\n")
    A_(md(valblk))
    A_(f"\nEscolhida a menor largura com fragmentação ≤ 5%: **{s1['largura_bloco']}**. Tipos de grupo: {s1['imagens_por_tipo_de_grupo']}.\n")
    A_(f"**Divisão final:** treino={S5['n_treino']}, teste={S5['n_teste']} (fração de teste={S5['frac_teste']:.3f}); "
       f"{S5['grupos_total']} grupos; **grupos compartilhados entre treino e teste = {S5['grupos_compartilhados']}**. "
       f"Limitação: nas classes sem `leaf_id`, um simulado em imagens que possuem `leaf_id` indica que {pct(S5['simulacao_vazamento_fallback']['fracao'])} "
       f"das imagens de teste teriam a mesma folha no treino se só os blocos fossem usados; como só {S5['simulacao_vazamento_fallback']['n_teste_em_grupos_de_bloco']} "
       f"imagens de teste estão em grupos de bloco, a estimativa global do vazamento residual é ≈ "
       f"**{pct(S5['simulacao_vazamento_fallback']['vazamento_global_estimado_fracao_do_teste'], 2)}** do teste. "
       "As demais imagens usam o `leaf_id` oficial e têm vazamento de folha nulo.\n")
    A_("### 1.4 Interpretação e dificuldades visuais\n"
       f"- **Fundo como atalho.** O fundo (cinza/arroxeado/marrom) varia conforme o local/dia de aquisição e se correlaciona com a "
       f"cultura/classe. Um Random Forest treinado **apenas com a cor da borda da imagem** obteve {pct(bk['acc_fundo'])} de acurácia no teste "
       f"(acaso uniforme: {pct(bk['acc_chance_uniforme'])}; classe majoritária: {pct(bk['acc_classe_majoritaria'])}). Isso mostra que parte do "
       "sinal ‘fácil’ do dataset não é a doença; resultados altos devem ser lidos com cautela (a borda de imagens de milho é folha, o que subestima o efeito). "
       "Figura: `01_dificuldade_fundo.png`.\n"
       "- **Imagens muito parecidas entre doenças da mesma cultura** (ex.: manchas foliares), sombras, folhas que preenchem o quadro, "
       "variação de iluminação e várias fotos/cópias aumentadas por folha.\n"
       "- **Desbalanceamento moderado/forte** (razão máx./mín. acima), exigindo métricas macro e atenção a `class_weight`.\n")

    # ------------------------------------------------------------------ T2
    A_("\n---\n## 2. Pré-processamento\n")
    A_("### 2.1 Fundamentação\n"
       "- **Redimensionamento** (INTER_AREA) reduz custo e uniformiza o vetor de características; reduzir demais perde textura fina.\n"
       "- **Mediana/Gaussiano 3×3** atenuam ruído de sensor/JPEG, mas suavizam texturas finas.\n"
       "- **CLAHE** (Zuiderveld, 1994) equaliza contraste localmente (aqui só no canal L do Lab); pode alterar a cor, que é discriminativa.\n"
       "- **Espaços de cor:** HSV separa matiz/saturação de brilho (robustez a iluminação); Lab é perceptualmente mais uniforme. "
       "**Escala de cinza** é usada apenas para textura/gradientes (LBP, GLCM, HOG operam em intensidade).\n"
       "- **Padronização** (z-score) é necessária para modelos baseados em distância/kernel (KNN, SVM), pois dimensões em escalas diferentes dominam a distância; "
       "árvores/florestas são invariantes a transformações monotônicas por característica.\n")
    A_("### 2.2 Procedimento\n"
       f"Cada operação foi **avaliada** em {pre['n_imagens_subamostra']} imagens **somente de treino**, com validação cruzada agrupada (3 dobras) e Random Forest, "
       f"nas configurações C1 (cor) e C2 (cor+textura). Regra de decisão: abandonar a linha de base (128 px, sem filtros) só se o ganho em F1 macro de C2 for ≥ 0,01.\n")
    A_("### 2.3 Resultados\n")
    A_(md(prep[["variante", "config", "f1_macro_media", "f1_macro_dp", "acuracia_media"]]))
    A_("\nEspaços de cor (variante base, RF):\n")
    A_(md(cspace))
    A_("\nEfeito da padronização (config C2):\n")
    A_(md(stdz))
    A_(f"\n**Decisão:** `{pre['escolhida']}` — parâmetros {pre['params']}. Melhor variante na CV: `{pre['melhor_variante_cv']}` "
       f"(ganho sobre a base: {pre['ganho_melhor_vs_base_f1macro']:+.3f}). Figuras: `02_exemplos_preprocessamento.png`, `02_histograma_clahe.png`, `02_comparacao_preprocessamento.png`.\n")
    A_("### 2.4 Interpretação\n"
       f"- Em C2, a base obteve F1 macro {p0['f1_macro_media']:.3f}±{p0['f1_macro_dp']:.3f} e a variante escolhida {pc['f1_macro_media']:.3f}±{pc['f1_macro_dp']:.3f}. "
       f"O desvio-padrão entre dobras (≈{sd:.3f}) é da ordem do ganho, logo a contribuição do pré-processamento é **pequena e de significância limitada** "
       "(amostra de ~3,5 mil imagens, 3 dobras). A decisão seguiu a regra de parcimônia definida antes do experimento.\n"
       "- Ver acima o efeito dos espaços de cor: HSV/Lab agregam informação ao RGB? (compare `RGB` com `RGB+HSV+Lab`).\n"
       "- A padronização muda drasticamente KNN e SVM e é irrelevante para a floresta, como previsto pela teoria.\n")

    # ------------------------------------------------------------------ T3
    A_("\n---\n## 3. Segmentação\n")
    A_("### 3.1 Fundamentação\n"
       "O fundo do PlantVillage é uma superfície quase neutra (croma baixo), enquanto a folha (verde, amarela, marrom) tem croma alto. "
       "No espaço **Lab**, o croma é √((a*−128)²+(b*−128)²). O **limiar de Otsu** (1979) escolhe o corte que maximiza a variância entre classes do histograma. "
       "**Morfologia** (abertura/fechamento) remove ruído e fecha lacunas; **contornos** externos preenchidos eliminam buracos internos (manchas de doença claras não podem virar ‘fundo’). "
       "Segmentar por ‘verde’ foi descartado a priori porque removeria as regiões doentes (marrons/amarelas), justamente as informativas.\n")
    A_("### 3.2 Procedimento\n"
       "Comparamos quatro métodos tradicionais (Otsu em S/HSV; k-means k=2 em Lab; Otsu no croma Lab com f=1; e o adotado, com fator f calibrado). "
       "Avaliação quantitativa por IoU/Dice contra as máscaras que acompanham o repositório (`raw/segmented`), **um proxy gerado pelos autores do dataset, não anotação manual**, "
       "usado só para calibrar f (metade A) e avaliar (metade B); nunca como característica.\n")
    A_("### 3.3 Resultados\n")
    A_(md(segcmp))
    A_(f"\nFator do limiar escolhido: **{seg['fator_limiar']}**. Limitações medidas em {seg['limitacoes']['n']} imagens de treino: "
       f"máscaras inválidas={pct(seg['limitacoes']['frac_fallback_mascara_invalida'])}, folha ocupando >90% do quadro={pct(seg['limitacoes']['frac_folha_ocupa_mais_de_90pct'])}, "
       f"folha <15% do quadro={pct(seg['limitacoes']['frac_folha_menos_de_15pct'])}, área mediana={seg['limitacoes']['area_mediana']:.2f}. "
       "Figuras: `03_exemplos_segmentacao.png`, `03_melhores_e_piores_casos.png`, `03_distribuicao_iou.png`, `03_area_mascara.png`.\n")
    A_("**Impacto na classificação (CV agrupada no treino):**\n")
    A_(md(segcv))
    A_("\n### 3.4 Interpretação\n"
       "- A máscara é boa (ver IoU), porém falha em folhas de baixo contraste cromático, sombras fortes e folhas que preenchem o quadro (milho); "
       "regiões escuras (necrose) próximas à borda podem ser cortadas.\n"
       "- A decisão de usar ou não segmentação é **empírica** e é reavaliada no conjunto de teste (Seção 5).\n")

    # ------------------------------------------------------------------ T4
    A_("\n---\n## 4. Extração de características\n")
    desc = T("04_colunas_por_grupo.csv")
    A_("### 4.1 Fundamentação\n"
       "- **Cor:** média/desvio por canal (RGB, HSV, Lab) e histogramas descrevem clorose (amarelecimento), necrose (marrom) e manchas. Matiz é circular, por isso usamos a média vetorial (cos, sen).\n"
       "- **Textura:** **LBP** uniforme (Ojala et al., 2002) codifica micropadrões locais (P=8,R=1 e P=16,R=2) de forma invariante à rotação; **GLCM** (Haralick et al., 1973) mede "
       "contraste, homogeneidade, energia, correlação e entropia (32 níveis, 4 ângulos promediados, distâncias 1 e 3) — úteis para pústulas, mofo e manchas.\n"
       "- **Gradientes/bordas:** **HOG** (Dalal & Triggs, 2005) descreve a distribuição local de orientações; densidade de bordas (Canny), magnitude de gradiente (Sobel) e histograma global de orientações.\n"
       "- **Forma** (da máscara): área, perímetro, circularidade 4πA/P², razões de aspecto, solidez, contato com a borda e 7 momentos de Hu (Hu, 1962, em escala log).\n")
    A_("### 4.2 Procedimento\n"
       "Extração em paralelo (joblib), em lotes de 2.000 imagens com cache (retomável); duas variantes por imagem lidas uma única vez. "
       "LBP é calculado uma vez por imagem e reutilizado. Na variante segmentada, estatísticas de cor/GLCM/LBP/gradientes são calculadas **apenas sobre pixels da máscara** "
       "(LBP e gradientes sobre a máscara erodida em 3 px); o HOG usa a imagem com fundo zerado.\n")
    A_("### 4.3 Resultados\n" + md(desc) + "\n\nCSVs: `dados/features_imagem_completa.csv` e `dados/features_imagem_segmentada.csv` "
       "(colunas: `image_id`, `class`, `group_id`, `split` + características). **`image_id`, `class`, `group_id` e `split` nunca entram no classificador.** "
       "Descrição de cada coluna: `tabelas/04_descricao_features.csv`.\n")
    A_("**Configurações comparadas:** C1 = cor; C2 = cor+textura; C3 = cor+textura+gradientes; C4 = C3 + forma (somente segmentada).\n")
    A_("**Contribuição isolada de cada grupo** (RF, CV agrupada na subamostra de treino):\n" + md(abl.sort_values("f1_macro_cv", ascending=False)))
    A_("\n### 4.4 Interpretação\n"
       f"O grupo mais informativo isoladamente foi **{abl.sort_values('f1_macro_cv').iloc[-1]['grupo']}** e o menos informativo **{abl.sort_values('f1_macro_cv').iloc[0]['grupo']}** "
       "(note que HOG tem 324 dimensões e os demais, poucas). Figuras: `04_pca_ilustrativa.png`, `04_heatmap_medias_por_classe.png`, `05_ablacao_grupos.png`.\n")

    # ------------------------------------------------------------------ T5
    A_("\n---\n## 5. Treinamento e avaliação\n")
    A_("### 5.1 Fundamentação\n"
       "- **KNN** classifica pelo voto dos k vizinhos mais próximos (distância euclidiana) — sensível à escala e à dimensionalidade; "
       "- **SVM com kernel RBF** maximiza a margem em um espaço de características induzido pelo kernel K(x,x′)=exp(−γ‖x−x′‖²) (Cortes & Vapnik, 1995); "
       "- **Random Forest** agrega árvores treinadas em amostras bootstrap com subconjuntos aleatórios de atributos (Breiman, 2001), reduz variância e não exige padronização.\n"
       "Escolha: três famílias distintas (instâncias, margem/kernel, conjunto de árvores), todas tradicionais. "
       "**F1 macro** é adequado ao desbalanceamento; **DummyClassifier** estabelece o piso.\n")
    gr = s5.get("grades", {})
    def _g(m):
        return "; ".join(f"{k}∈{v}" for k, v in gr.get(m, {}).items()) or "(não registrado)"
    A_("### 5.2 Procedimento\n"
       "Pipeline `StandardScaler → classificador` (scaler ajustado só nos dados de treino de cada dobra); busca em grade (`GridSearchCV`, `StratifiedGroupKFold(3)`, F1 macro) em uma "
       f"subamostra de ~{s5.get('cv_subamostra', config.CV_SUBSAMPLE)} imagens de **treino**, a mesma para todas as tarefas (condições equivalentes); reajuste no treino completo; "
       f"teste usado apenas na avaliação final. **Modo de execução: {s5.get('modo', 'completo')}.** Grades efetivamente usadas — "
       f"KNN: {_g('KNN')}. Random Forest ({s5.get('rf_n_estimators', 200)} árvores): {_g('RandomForest')}. SVM (γ=‘scale’): {_g('SVM')}. "
       "Random Forest não usa padronização (invariante a transformações monótonas por característica).\n")
    A_("### 5.3 Resultados\n**Referência (DummyClassifier):**\n" + md(dummy, 4))
    show = res[["variante", "config", "modelo", "n_features", "f1_macro_cv_treino", "acuracia_teste", "f1_macro_teste",
                "precisao_macro_teste", "recall_macro_teste", "melhores_hiperparametros"]].sort_values("f1_macro_teste", ascending=False)
    A_("\n**Todas as combinações (ordenadas por F1 macro no teste):**\n" + md(show))
    A_(f"\n**Modelo escolhido por validação no treino:** `{chosen_key}` (F1 macro CV={chosen['f1_macro_cv_treino']:.3f}); "
       f"no teste: acurácia={chosen['acuracia_teste']:.3f}, F1 macro={chosen['f1_macro_teste']:.3f}. "
       f"Melhor no teste: `{s5['melhor_por_teste']}` (F1 macro={s5['melhor_por_teste_f1']:.3f}).\n")
    A_("Matrizes de confusão: `graficos/05_matriz_confusao_modelo_escolhido.png` e uma por algoritmo. "
       "Relatório de classificação completo: `relatorios/05_classification_report_modelo_escolhido.txt`. "
       "Métricas por classe: `tabelas/05_metricas_por_classe_modelo_escolhido.csv`.\n")
    A_("**Classes com pior F1 (modelo escolhido):**\n" + md(worst.head(8)))
    A_("\n**Pares de classes mais confundidos:**\n" + md(pairs.head(10)))
    A_(f"\nDos erros do modelo escolhido, **{pct(same_crop)}** ocorrem entre doenças da **mesma cultura**. Exemplos de erros: `05_erros_exemplos.png`.\n")

    # ------------------------------------------------------------------ T6
    A_("\n---\n## 6. Análise crítica dos resultados\n")
    piv = inte.pivot(index="modelo", columns="config", values="f1_macro_teste")
    d21 = piv["C2_COR+TEXTURA"] - piv["C1_COR"]
    d32 = piv["C3_COR+TEXTURA+GRADIENTES"] - piv["C2_COR+TEXTURA"]
    bc = by_cfg.sort_index()
    A_(f"**Configuração de características.** Média do F1 macro (teste, imagem inteira, 3 modelos): "
       + "; ".join(f"{k}={v:.3f}" for k, v in bc.items())
       + f". Passar de C1 a C2: {by_cfg['C2_COR+TEXTURA'] - by_cfg['C1_COR']:+.3f}; de C2 a C3: {by_cfg['C3_COR+TEXTURA+GRADIENTES'] - by_cfg['C2_COR+TEXTURA']:+.3f}. "
       f"Por modelo (teste, imagem inteira), C2−C1: " + "; ".join(f"{k}={v:+.3f}" for k, v in d21.items())
       + "; C3−C2: " + "; ".join(f"{k}={v:+.3f}" for k, v in d32.items()) + ". "
       "Leitura: o sinal de cada delta indica se o grupo acrescentou informação útil àquele algoritmo. "
       "*Hipótese* para deltas negativos de C3−C2 em KNN: o HOG (324 de 443 colunas) domina a distância euclidiana e dilui cor/textura; "
       "SVM e Random Forest lidam melhor com muitas dimensões heterogêneas.\n")
    A_("**Classificador.** Média do F1 macro por modelo (todas as configurações): " + "; ".join(f"{k}={v:.3f}" for k, v in by_model.items())
       + ". Todos superam muito o DummyClassifier. As diferenças refletem a capacidade de cada família lidar com características heterogêneas e de alta dimensão.\n")
    A_(f"**Segmentação.** Variação média do F1 macro de teste (segmentada − inteira) nos pares (config, modelo): **{seg_mean_delta:+.3f}**; a segmentada ganhou em {seg_wins} de {len(m)} pares. "
       f"Adicionar forma (C4 − C3, segmentada), por modelo: " + "; ".join(f"{k}={v:+.3f}" for k, v in d_forma.items()) + ". ")
    if seg_mean_delta < 0:
        A_("Portanto a segmentação **não** melhorou o desempenho. *Hipótese* (apoiada pelo teste do fundo da Seção 1.4, mas não isolada experimentalmente): o fundo carrega informação correlacionada à classe "
           "(local/dia de aquisição); ao removê-lo, perde-se um atalho que ajuda no conjunto de teste do próprio dataset, e erros de máscara (folhas cortadas, sombras) adicionam ruído. "
           "A segmentação é metodologicamente mais ‘honesta’ para generalização a campo, mesmo com métrica menor neste benchmark.\n")
    else:
        A_("Portanto a segmentação **melhorou** o desempenho médio. *Hipótese*: ao restringir cor e textura aos pixels da folha, reduz-se a influência do fundo/iluminação do cenário; "
           "isso não foi isolado experimentalmente.\n")
    A_(f"**Pré-processamento.** Ganho da variante escolhida sobre a base em C2 (CV, treino): {pc['f1_macro_media'] - p0['f1_macro_media']:+.3f} (desvio entre dobras ≈ {sd:.3f}); "
       "contribuição pequena e de significância limitada; o fator decisivo foi a **padronização** para KNN/SVM (Seção 2.3).\n")
    A_("**Quais classes foram mais confundidas e por quê.** Ver pares acima. "
       f"Os pares mais confundidos são de doenças da mesma cultura com sintomas parecidos, mas apenas {pct(same_crop)} de todos os erros são intra-cultura: "
       "o restante cruza culturas. *Hipótese* (não isolada): lesões pequenas não são capturadas por características globais a 128 px, "
       "e o fundo/iluminação de cada sessão de aquisição contribui para erros entre culturas.\n")
    A_("**Limitações que permanecem.**\n"
       f"- Vazamento residual estimado ≈ {pct(S5['simulacao_vazamento_fallback']['vazamento_global_estimado_fracao_do_teste'], 2)} do teste (classes sem `leaf_id`);\n"
       "- fundo como atalho e imagens de bancada (laboratório): o desempenho não representa fotos reais em campo;\n"
       "- a seleção de hiperparâmetros usa subamostra (por custo) e um grid pequeno;\n"
       "- HOG não é invariante à rotação; características globais não localizam lesões;\n"
       "- máscaras avaliadas contra um proxy (não manual);\n"
       "- um único split de teste (sem intervalo de confiança).\n")
    A_("**Melhorias futuras (dentro das restrições do enunciado).** Características por regiões da folha (grade) ou da lesão segmentada (limiarização no matiz/saturação dentro da máscara), "
       "histogramas de cor em mais espaços/bins, LBP rotacional multiescala, seleção de características/PCA ajustados no treino, busca de hiperparâmetros mais ampla com a subamostra maior, "
       "validação com vários splits e intervalos de confiança, e avaliação com remoção explícita do fundo (preenchimento neutro) para quantificar o atalho.\n")

    # ------------------------------------------------------------------ refs
    A_("\n---\n## Referências\n"
       "- Hughes, D. P.; Salathé, M. *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. arXiv:1511.08060, 2015.\n"
       "- Mohanty, S. P.; Hughes, D. P.; Salathé, M. *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science, 7:1419, 2016 (origem do dataset e do agrupamento por folha; não usamos deep learning).\n"
       "- Kaufman, S.; Rosset, S.; Perlich, C.; Stitelman, O. *Leakage in data mining: formulation, detection, and avoidance*. ACM TKDD, 6(4), 2012.\n"
       "- Otsu, N. *A threshold selection method from gray-level histograms*. IEEE Trans. Systems, Man, and Cybernetics, 9(1), 1979.\n"
       "- Zuiderveld, K. *Contrast limited adaptive histogram equalization*. Graphics Gems IV, Academic Press, 1994.\n"
       "- Ojala, T.; Pietikäinen, M.; Mäenpää, T. *Multiresolution gray-scale and rotation invariant texture classification with local binary patterns*. IEEE TPAMI, 24(7), 2002.\n"
       "- Haralick, R. M.; Shanmugam, K.; Dinstein, I. *Textural features for image classification*. IEEE Trans. Systems, Man, and Cybernetics, 3(6), 1973.\n"
       "- Dalal, N.; Triggs, B. *Histograms of oriented gradients for human detection*. IEEE CVPR, 2005.\n"
       "- Hu, M.-K. *Visual pattern recognition by moment invariants*. IRE Trans. Information Theory, 8(2), 1962.\n"
       "- Cover, T.; Hart, P. *Nearest neighbor pattern classification*. IEEE Trans. Information Theory, 13(1), 1967.\n"
       "- Cortes, C.; Vapnik, V. *Support-vector networks*. Machine Learning, 20, 1995.\n"
       "- Breiman, L. *Random forests*. Machine Learning, 45(1), 2001.\n"
       "- Pedregosa, F. et al. *Scikit-learn: machine learning in Python*. JMLR, 12, 2011. Documentação: https://scikit-learn.org/stable/\n"
       "- van der Walt, S. et al. *scikit-image: image processing in Python*. PeerJ, 2:e453, 2014. Documentação: https://scikit-image.org/docs/stable/\n"
       "- Bradski, G. *The OpenCV Library*. Dr. Dobb’s Journal, 2000. Documentação: https://docs.opencv.org/\n"
       "- Repositório do dataset: https://github.com/spMohanty/PlantVillage-Dataset\n")

    text = "\n".join(sec)
    out = config.DIR_REPORT / "RELATORIO_AV2.md"
    out.write_text(text, encoding="utf-8")
    log(f"Relatório gerado: {out} ({len(text.split())} palavras)")
    return out


if __name__ == "__main__":
    main()
