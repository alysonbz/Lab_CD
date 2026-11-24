
from analise_dataset.config import prepare_directories

from analise_dataset import (
    dataset,
    exploracao,
    limpeza,
    divisao,
    preprocessamento,
    segmentacao,
    features,
    treinamento,
    analise_resultados
)


def main():

    prepare_directories()

    etapas = [
        ("Download do dataset", dataset.run),
        ("Análise exploratória", exploracao.run),
        ("Limpeza dos dados", limpeza.run),
        ("Divisão treino e teste", divisao.run),
        ("Pré-processamento", preprocessamento.run),
        ("Segmentação", segmentacao.run),
        ("Extração de características", features.run),
        ("Treinamento e avaliação", treinamento.run),
        ("Análise dos resultados", analise_resultados.run)
    ]

    for nome, funcao in etapas:

        print("\n" + "=" * 70)
        print(nome.upper())
        print("=" * 70)

        funcao()

    print("\nPROJETO CONCLUÍDO!")
    print("Confira os resultados na pasta outputs/")


if __name__ == "__main__":
    main()
