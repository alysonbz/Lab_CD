
import json
import random
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from .config import (
    OUTPUT_DIR,
    DATA_DIR,
    CLASSES,
    RANDOM_STATE
)

from .features import construir_dataframe


# --------------------------------------------------
# 1. DATA AUGMENTATION
# --------------------------------------------------

def aumentar_imagem(caminho, rng):
    imagem = cv2.imread(str(caminho))

    if imagem is None:
        raise ValueError(f"Erro ao abrir imagem: {caminho}")

    h, w = imagem.shape[:2]

    # Rotação moderada
    angulo = rng.uniform(-20, 20)

    matriz = cv2.getRotationMatrix2D(
        (w // 2, h // 2),
        angulo,
        rng.uniform(0.90, 1.10)
    )

    matriz[0, 2] += rng.uniform(-0.05, 0.05) * w
    matriz[1, 2] += rng.uniform(-0.05, 0.05) * h

    aumentada = cv2.warpAffine(
        imagem,
        matriz,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )

    # Espelhamentos
    if rng.random() < 0.5:
        aumentada = cv2.flip(aumentada, 1)

    if rng.random() < 0.15:
        aumentada = cv2.flip(aumentada, 0)

    # Brilho e contraste
    alpha = rng.uniform(0.90, 1.10)
    beta = rng.uniform(-12, 12)

    aumentada = cv2.convertScaleAbs(
        aumentada,
        alpha=alpha,
        beta=beta
    )

    return aumentada


def balancear_treinamento(registros, identificador="treino"):
    rng = random.Random(RANDOM_STATE)

    pasta_base = DATA_DIR / "dataset_aumentado"
    pasta = pasta_base / identificador

    # Remove somente a pasta desta etapa
    if pasta.exists():
        shutil.rmtree(pasta)

    pasta.mkdir(parents=True, exist_ok=True)

    contagens = {
        classe: sum(
            1 for r in registros
            if r["label"] == classe
        )
        for classe in CLASSES
    }

    if any(quantidade == 0 for quantidade in contagens.values()):
        raise ValueError(
            f"Há classes sem imagens no treinamento: {contagens}"
        )

    quantidade_alvo = max(contagens.values())
    novos_registros = []

    for classe in CLASSES:
        originais = [
            r for r in registros
            if r["label"] == classe
        ]

        destino = pasta / classe
        destino.mkdir(parents=True, exist_ok=True)

        # Copia as imagens originais
        for registro in originais:
            caminho = Path(registro["path"])

            imagem = cv2.imread(str(caminho))

            if imagem is None:
                raise ValueError(
                    f"Não foi possível ler a imagem: {caminho}"
                )

            novo_caminho = destino / caminho.name

            sucesso = cv2.imwrite(
                str(novo_caminho),
                imagem
            )

            if not sucesso:
                raise IOError(
                    f"Não foi possível salvar a imagem: {novo_caminho}"
                )

            novos_registros.append({
                "path": str(novo_caminho.resolve()),
                "label": classe
            })

        faltantes = quantidade_alvo - len(originais)

        # Aumenta somente as classes minoritárias
        for i in range(faltantes):
            original = rng.choice(originais)
            caminho_original = Path(original["path"])

            imagem = aumentar_imagem(
                str(caminho_original),
                rng
            )

            nome = f"aug_{i}_{caminho_original.stem}.jpg"
            novo_caminho = destino / nome

            sucesso = cv2.imwrite(
                str(novo_caminho),
                imagem
            )

            if not sucesso:
                raise IOError(
                    f"Não foi possível salvar a imagem aumentada: "
                    f"{novo_caminho}"
                )

            novos_registros.append({
                "path": str(novo_caminho.resolve()),
                "label": classe
            })

        print(
            f"{classe}: {len(originais)} -> {quantidade_alvo}"
        )

    return novos_registros


# --------------------------------------------------
# 2. CLASSIFICADORES
# --------------------------------------------------

def criar_modelos():
    modelos = {
        "Baseline": [
            DummyClassifier(strategy="most_frequent")
        ],

        "KNN": [
            KNeighborsClassifier(n_neighbors=5),
            KNeighborsClassifier(n_neighbors=7),
            KNeighborsClassifier(n_neighbors=11)
        ],

        "SVM": [
            SVC(kernel="linear", C=0.1),
            SVC(kernel="linear", C=1),
            SVC(kernel="rbf", C=1, gamma="scale"),
            SVC(kernel="rbf", C=10, gamma="scale")
        ],

        "Random Forest": [
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            ),
            RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced"
            )
        ]
    }

    return modelos


def criar_pipeline(modelo, escalar=True):
    etapas = [
        ("imputacao", SimpleImputer(strategy="median"))
    ]

    if escalar:
        etapas.append(
            ("padronizacao", StandardScaler())
        )

    etapas.append(("classificador", modelo))

    return Pipeline(etapas)


# --------------------------------------------------
# 3. PREPARAÇÃO DOS DADOS
# --------------------------------------------------

def preparar_dados(registros_treino, registros_teste):
    treino, validacao = train_test_split(
        registros_treino,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=[r["label"] for r in registros_treino]
    )

    treino_balanceado = balancear_treinamento(
        treino,
        identificador="treino_validacao"
    )

    return treino_balanceado, validacao, registros_teste


# --------------------------------------------------
# 4. AVALIAÇÃO
# --------------------------------------------------

def avaliar_modelo(modelo, X, y):
    predicoes = modelo.predict(X)

    return {
        "acuracia": accuracy_score(y, predicoes),
        "precisao_macro": precision_score(
            y, predicoes, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y, predicoes, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y, predicoes, average="macro", zero_division=0
        )
    }


# --------------------------------------------------
# 5. TREINAMENTO E SELEÇÃO
# --------------------------------------------------

def executar_experimento(configuracao):
    with open(
        OUTPUT_DIR / "split.json",
        "r",
        encoding="utf-8"
    ) as arquivo:
        dados = json.load(arquivo)

    treino, validacao, teste = preparar_dados(
        dados["train"],
        dados["test"]
    )

    print("\nExtraindo características...")

    df_treino = construir_dataframe(
        treino, configuracao
    )

    df_validacao = construir_dataframe(
        validacao, configuracao
    )

    df_teste = construir_dataframe(
        teste, configuracao
    )

    colunas_excluidas = [
        "classe",
        "imagem_id",
        "caminho"
    ]

    X_treino = df_treino.drop(
        columns=colunas_excluidas
    )

    y_treino = df_treino["classe"]

    X_validacao = df_validacao.drop(
        columns=colunas_excluidas
    )

    y_validacao = df_validacao["classe"]

    X_teste = df_teste.drop(
        columns=colunas_excluidas
    )

    y_teste = df_teste["classe"]

    # Garante mesmas colunas
    X_validacao = X_validacao.reindex(
        columns=X_treino.columns
    )

    X_teste = X_teste.reindex(
        columns=X_treino.columns
    )

    modelos = criar_modelos()

    resultados_validacao = []

    melhor_f1 = -1
    melhor_nome = None
    melhor_modelo = None

    for nome, configuracoes in modelos.items():
        for modelo_base in configuracoes:

            escalar = nome in ["KNN", "SVM"]

            pipeline = criar_pipeline(
                modelo_base,
                escalar
            )

            pipeline.fit(
                X_treino,
                y_treino
            )

            metricas = avaliar_modelo(
                pipeline,
                X_validacao,
                y_validacao
            )

            resultados_validacao.append({
                "configuracao": configuracao,
                "modelo": nome,
                "parametros": str(
                    modelo_base.get_params()
                ),
                **metricas
            })

            if metricas["f1_macro"] > melhor_f1:
                melhor_f1 = metricas["f1_macro"]
                melhor_nome = nome
                melhor_modelo = modelo_base

    # --------------------------------------------------
    # 6. REAJUSTE COM TODO O TREINAMENTO
    # --------------------------------------------------

    treino_final = balancear_treinamento(
        dados["train"],
        identificador=f"treino_final_{configuracao}"
    )

    df_treino_final = construir_dataframe(
        treino_final,
        configuracao
    )

    X_treino_final = df_treino_final.drop(
        columns=colunas_excluidas
    )

    y_treino_final = df_treino_final["classe"]

    X_treino_final = X_treino_final.reindex(
        columns=X_treino.columns
    )

    modelo_final = criar_pipeline(
        melhor_modelo,
        melhor_nome in ["KNN", "SVM"]
    )

    modelo_final.fit(
        X_treino_final,
        y_treino_final
    )

    # --------------------------------------------------
    # 7. TESTE FINAL
    # --------------------------------------------------

    metricas_teste = avaliar_modelo(
        modelo_final,
        X_teste,
        y_teste
    )

    predicoes = modelo_final.predict(X_teste)

    print("\nModelo escolhido:", melhor_nome)
    print("Configuração:", configuracao)

    print("\nResultado no teste:")
    print(
        classification_report(
            y_teste,
            predicoes,
            labels=CLASSES,
            zero_division=0
        )
    )

    # Relatório detalhado por classe
    relatorio = classification_report(
        y_teste,
        predicoes,
        labels=CLASSES,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(relatorio).transpose().to_csv(
        OUTPUT_DIR /
        f"relatorio_{configuracao}.csv"
    )

    # Matriz de confusão
    matriz = confusion_matrix(
        y_teste,
        predicoes,
        labels=CLASSES
    )

    fig, ax = plt.subplots(figsize=(9, 7))

    ConfusionMatrixDisplay(
        matriz,
        display_labels=CLASSES
    ).plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=35,
        values_format="d"
    )

    ax.set_title(
        f"Matriz de confusão - {configuracao}"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        f"matriz_confusao_{configuracao}.png",
        dpi=150
    )

    plt.close(fig)

    # Exemplos classificados incorretamente
    erros = []

    for caminho, real, previsto in zip(
        df_teste["caminho"],
        y_teste,
        predicoes
    ):
        if real != previsto:
            erros.append({
                "imagem_id": Path(caminho).name,
                "caminho": caminho,
                "classe_real": real,
                "classe_prevista": previsto
            })

    pd.DataFrame(erros).to_csv(
        OUTPUT_DIR /
        f"erros_{configuracao}.csv",
        index=False
    )

    return {
        "configuracao": configuracao,
        "modelo": melhor_nome,
        **metricas_teste
    }, resultados_validacao


def run():
    resultados = []
    validacoes = []

    for configuracao in [
        "cor",
        "cor_textura",
        "completa"
    ]:

        print("\n" + "=" * 60)
        print("EXPERIMENTO:", configuracao.upper())
        print("=" * 60)

        resultado, validacao = executar_experimento(
            configuracao
        )

        resultados.append(resultado)
        validacoes.extend(validacao)

    pd.DataFrame(resultados).to_csv(
        OUTPUT_DIR / "metricas_modelos.csv",
        index=False
    )

    pd.DataFrame(validacoes).to_csv(
        OUTPUT_DIR / "metricas_validacao.csv",
        index=False
    )

    print("\nExperimentos concluídos!")


if __name__ == "__main__":
    run()
