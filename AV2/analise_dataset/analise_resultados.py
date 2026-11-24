
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .config import OUTPUT_DIR, CLASSES


def comparar_modelos():

    arquivo = OUTPUT_DIR / "metricas_modelos.csv"

    df = pd.read_csv(arquivo)

    print("\nCOMPARAÇÃO DOS MODELOS")
    print(df.to_string(index=False))

    # Ordena pelo F1 macro
    df = df.sort_values(
        "f1_macro",
        ascending=False
    )

    melhor = df.iloc[0]

    print("\nMELHOR CONFIGURAÇÃO")
    print(melhor)

    # Gráfico de F1 macro
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = (
        df["configuracao"] + " - " + df["modelo"]
    )

    ax.barh(
        labels,
        df["f1_macro"]
    )

    ax.set_xlabel("F1 Macro")
    ax.set_title("Comparação dos classificadores")
    ax.grid(axis="x", alpha=0.3)

    ax.invert_yaxis()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "comparacao_modelos.png",
        dpi=150
    )

    plt.close(fig)

    # Comparação de acurácia
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        labels,
        df["acuracia"]
    )

    ax.set_xlabel("Acurácia")
    ax.set_title("Acurácia dos classificadores")
    ax.grid(axis="x", alpha=0.3)

    ax.invert_yaxis()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "comparacao_acuracia.png",
        dpi=150
    )

    plt.close(fig)

    return df, melhor


def analisar_classes(configuracao):

    arquivo = (
        OUTPUT_DIR /
        f"relatorio_{configuracao}.csv"
    )

    df = pd.read_csv(arquivo, index_col=0)

    resultados = []

    for classe in CLASSES:

        if classe not in df.index:
            continue

        resultados.append({
            "classe": classe,
            "precisao": df.loc[classe, "precision"],
            "recall": df.loc[classe, "recall"],
            "f1_score": df.loc[classe, "f1-score"],
            "suporte": df.loc[classe, "support"]
        })

    resultado = pd.DataFrame(resultados)

    resultado = resultado.sort_values(
        "f1_score"
    )

    print("\nDESEMPENHO POR CLASSE")
    print(resultado.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        resultado["classe"],
        resultado["f1_score"]
    )

    ax.set_ylim(0, 1)
    ax.set_ylabel("F1-score")
    ax.set_title(
        f"Desempenho por classe - {configuracao}"
    )

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR /
        f"desempenho_classes_{configuracao}.png",
        dpi=150
    )

    plt.close(fig)

    return resultado


def analisar_erros(configuracao):

    arquivo = (
        OUTPUT_DIR /
        f"erros_{configuracao}.csv"
    )

    df = pd.read_csv(arquivo)

    print("\nANÁLISE DOS ERROS")

    print("Total de erros:", len(df))

    if df.empty:
        print("Nenhum erro encontrado.")
        return df

    confusoes = (
        df.groupby(
            ["classe_real", "classe_prevista"]
        )
        .size()
        .reset_index(name="quantidade")
        .sort_values(
            "quantidade",
            ascending=False
        )
    )

    print("\nPrincipais confusões:")
    print(confusoes.head(10).to_string(index=False))

    confusoes.to_csv(
        OUTPUT_DIR /
        f"confusoes_{configuracao}.csv",
        index=False
    )

    return df


def gerar_relatorio():

    resultados, melhor = comparar_modelos()

    configuracao = melhor["configuracao"]

    desempenho = analisar_classes(configuracao)

    erros = analisar_erros(configuracao)

    texto = []

    texto.append(
        "# Relatório de classificação de resíduos\n"
    )

    texto.append("## 1. Metodologia\n")

    texto.append(
        "Foram utilizadas características tradicionais de "
        "processamento digital de imagens, sem redes neurais "
        "ou modelos pré-treinados.\n"
    )

    texto.append(
        "Foram comparadas três configurações: características "
        "de cor, cor combinada com textura e configuração "
        "completa, incluindo forma e gradientes.\n"
    )

    texto.append(
        "O balanceamento foi realizado exclusivamente "
        "no conjunto de treinamento por Data Augmentation.\n"
    )

    texto.append("## 2. Comparação dos modelos\n")

    texto.append(
        resultados.to_markdown(index=False)
    )

    texto.append("\n## 3. Melhor resultado\n")

    texto.append(
        f"**Classificador:** {melhor['modelo']}\n"
    )

    texto.append(
        f"**Características:** {configuracao}\n"
    )

    texto.append(
        f"**Acurácia:** {melhor['acuracia']:.4f}\n"
    )

    texto.append(
        f"**F1 Macro:** {melhor['f1_macro']:.4f}\n"
    )

    texto.append("## 4. Desempenho por classe\n")

    texto.append(
        desempenho.to_markdown(index=False)
    )

    texto.append("\n## 5. Análise dos erros\n")

    texto.append(
        f"Foram identificados {len(erros)} erros "
        "no conjunto de teste.\n"
    )

    if not erros.empty:

        confusoes = (
            erros.groupby(
                ["classe_real", "classe_prevista"]
            )
            .size()
            .sort_values(ascending=False)
        )

        for (real, previsto), quantidade in confusoes.head(10).items():

            texto.append(
                f"- {real} confundido com {previsto}: "
                f"{quantidade} ocorrência(s).\n"
            )

    texto.append("\n## 6. Discussão\n")

    texto.append(
        "As características de cor podem ajudar a distinguir "
        "materiais com propriedades visuais diferentes. "
        "Entretanto, vidro, plástico e metal podem apresentar "
        "reflexos e cores semelhantes dependendo da iluminação.\n"
    )

    texto.append(
        "As características de textura podem auxiliar na "
        "distinção de superfícies lisas, rugosas, amassadas "
        "ou com padrões impressos.\n"
    )

    texto.append(
        "As características de forma podem contribuir na "
        "identificação de garrafas, caixas e embalagens. "
        "Entretanto, sua confiabilidade depende da qualidade "
        "da segmentação do objeto.\n"
    )

    texto.append(
        "A utilização de Data Augmentation aumenta a diversidade "
        "das imagens de treinamento, mas não substitui a coleta "
        "de novos exemplos reais.\n"
    )

    texto.append(
        "Como o dataset não disponibiliza identificadores "
        "de objetos físicos, não é possível garantir que "
        "fotografias do mesmo objeto estejam em partições "
        "diferentes. Essa limitação deve ser considerada "
        "na interpretação dos resultados.\n"
    )

    texto.append("\n## 7. Conclusão\n")

    texto.append(
        f"A melhor configuração experimental foi "
        f"{configuracao}, utilizando {melhor['modelo']}, "
        f"com F1 Macro de {melhor['f1_macro']:.4f}. "
        "A escolha deve ser interpretada considerando "
        "as limitações do dataset e a análise das classes "
        "com maior índice de confusão.\n"
    )

    caminho = OUTPUT_DIR / "relatorio_resultados.md"

    caminho.write_text(
        "\n".join(texto),
        encoding="utf-8"
    )

    print("\nRelatório salvo em:", caminho)


def run():

    gerar_relatorio()

    print("\nAnálise de resultados concluída!")


if __name__ == "__main__":
    run()
