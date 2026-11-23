from analise_dataset.config import prepare_directories
from analise_dataset import dataset, exploracao, limpeza, divisao, preprocessamento, segmentacao

def main():
    prepare_directories()
    etapas = [
        ("Download e inspeção do dataset", dataset.run),
        ("Análise exploratória", exploracao.run),
        ("Verificação e limpeza", limpeza.run),
        ("Divisão treino/teste", divisao.run),
        ("Pré-processamento", preprocessamento.run),
        ("Segmentação", segmentacao.run),
    ]
    for title, action in etapas:
        print("\\n" + "=" * 70)
        print(title.upper())
        print("=" * 70)
        action()
    print("\\nFluxo concluído. Confira os resultados na pasta outputs/.")

if __name__ == "__main__":
    main()
