####################################################################################################
#------------------------------ETAPA 01 - Análise e Preparação do dataset -------------------------#
####################################################################################################
# Importações 
from pathlib import Path
from collections import Counter
import pandas as pd
import cv2
import hashlib
import random
import matplotlib.pyplot as plt
import numpy as np
import time



print("="*50)
print("ETAPA 01 - ANÁLISE E PREPARAÇÃO DO DATASET")
print("="*50)
print("1.1 - Identificação das classes e quantidade de imagens por classe")
print("="*50)

print(Path.cwd())
caminho_dataset = Path(r"fruits-360-100x100")
caminho_treino = caminho_dataset / "Training"
caminho_teste = caminho_dataset / "Test"

print(caminho_dataset.exists())
print(caminho_treino.exists())
print(caminho_teste.exists())


# ---- IDENTIFICAÇÃO DAS CLASSES ----

classes_treino = []
classes_teste = []

for pasta in caminho_treino.iterdir():
    if pasta.is_dir():
        print(pasta.name)
        classes_treino.append(pasta.name)

for pasta in caminho_teste.iterdir():
    if pasta.is_dir():
        #print(pasta.name)
        classes_teste.append(pasta.name)

comparacao = set(classes_treino).difference(set(classes_teste))
comparacao_inversa = set(classes_teste).difference(set(classes_treino))


print(f"Quantidade de classes em treino: {len(classes_treino)}")
print(f"\nQuantidade de classes em teste: {len(classes_teste)}")
print(f"\nClasses faltantes no conjunto de teste com base no treino:\n{comparacao}")
print(f"\nClasses faltantes no conjunto de treino com base no teste:\n{comparacao_inversa}")


classes_treino_normalizadas = [classe.casefold() for classe in classes_treino]
classes_teste_normalizadas = [classe.casefold() for classe in classes_teste]

comparacao = set(classes_treino_normalizadas).difference(set(classes_teste_normalizadas))
comparacao_inversa = set(classes_teste_normalizadas).difference(set(classes_treino_normalizadas))

print(f"\nClasses faltantes no conjunto de teste com base no treino:\n{comparacao}")
print(f"\nClasses faltantes no conjunto de treino com base no teste:\n{comparacao_inversa}")


# ---- CONTAGEM DAS IMAGENS POR CLASSE ----

pasta = caminho_treino / "Apple Braeburn 1"
imagens = list(pasta.glob("*.jpg"))
print(pasta.exists())
print(f"Quantidade de imagens da classe Apple Braeburn 1:\n{len(imagens)}")


# ---- CONTAGEM DAS IMAGENS POR CLASSE ----
dados_treino = []
dados_teste = []

def contagem_imagens(caminho, dados):
    for pasta in caminho.iterdir():
        if pasta.is_dir():
            imagens = list(pasta.glob("*.jpg"))
            quantidade = len(imagens)
            registro = {"classe": pasta.name, 
                        "quantidade": quantidade}
            dados.append(registro)
    return dados

df_treino = pd.DataFrame(contagem_imagens(caminho_treino, dados_treino))
df_teste = pd.DataFrame(contagem_imagens(caminho_teste, dados_teste))
print(f"---- Contagem das imagens por classe do conjunto de treino----")
df_treino.head()


print(f"---- Contagem das imagens por classe do conjunto de teste----\n")
df_teste.head()


print(50*("="))
print("------- INORMAÇÕES DO CONJUNTO DE TREINO ---------")
print(50*("="))

print(f"\nTamanho do df:{df_treino.shape}")
print(f"\n====== Informações: ======")
df_treino.info()
print(f"\n====== Total de imagens: =======")
print(df_treino['quantidade'].sum())
print(f"\n====== Estatísticas: ======")
df_treino.describe()


print(50*("="))
print("------- INORMAÇÕES DO CONJUNTO DE TESTE ---------")
print(50*("="))

print(f"\nTamanho:{df_teste.shape}")
print(f"\n====== Informações: ======")
df_teste.info()
print(f"\n====== Total de imagens: =======")
print(df_teste['quantidade'].sum())
print(f"\n====== Estatísticas: ======")
df_treino.describe()

# Analisando se há alguma classe sem registro
df_treino.loc[df_treino["quantidade"] == 0]
df_teste.loc[df_teste["quantidade"] == 0]

########################################################################################
# 1.2 - Verificação da integridade dos dados
#########################################################################################

print("="*50)
print("1.2 - Verificação da integridade dos dados")
print("="*50)

# Identificando possíveis erros ao carregar as imagens - OpenCV
def CarregarImagem_CV2(caminho):
    try:
        imagem_opencv = cv2.imread(str(caminho))
        if imagem_opencv is None:
            return False
        else:
            return True
    except Exception:
        return False

## Verificação do conjunto de treinamento
resultados_CV2_treino = []

for pasta in caminho_treino.iterdir():
    if pasta.is_dir():
        for caminho_imagem in pasta.glob("*.jpg"):
            resultado = CarregarImagem_CV2(caminho_imagem)
            registro = {"classe": pasta.name, 
                        "imagem": caminho_imagem.name,
                        "leitura_sucesso": resultado}
            resultados_CV2_treino.append(registro)


## Verificação do conjunto de teste
resultados_CV2_teste = []

for pasta in caminho_teste.iterdir():
    if pasta.is_dir():
        for caminho_imagem in pasta.glob("*.jpg"):
            resultado = CarregarImagem_CV2(caminho_imagem)
            registro = {"classe": pasta.name, 
                        "imagem": caminho_imagem.name, 
                        "leitura_sucesso": resultado}
            resultados_CV2_teste.append(registro)            


# Validando os resultados
print("="*50)
print(f"Imagens analisadas no treinamento: {len(resultados_CV2_treino)}")
print(f"Imagens analisadas no teste: {len(resultados_CV2_teste)}")
print(f"Resultados do treino: \n{resultados_CV2_treino[:5]}")
print(f"Resultados do teste: \n{resultados_CV2_teste[:5]}")
print("="*50)


# Armazenando os resultados em dataframes
df_resultados_CV2_treino = pd.DataFrame(resultados_CV2_treino)
df_resultados_CV2_teste = pd.DataFrame(resultados_CV2_teste)

# Verificando a estrutura dos Dataframes
df_resultados_CV2_treino.head()
print("="*50)
print(f"Tamanho do df de treino: {df_resultados_CV2_treino.shape}")
print("="*50)
print(df_resultados_CV2_treino.info())
print("="*50)


df_resultados_CV2_teste.head()
print("="*50)
print(f"Tamanho do df de teste: {df_resultados_CV2_teste.shape}")
print("="*50)
print(df_resultados_CV2_teste.info())
print("="*50)

# Imagens carregadas corretementeno treinamento
print("="*50)
sucesso_treino = df_resultados_CV2_treino["leitura_sucesso"].sum()
print(f"Imagens carregadas corretemente no treinamento: {sucesso_treino}")

# Imagens carregadas corretementeno teste
sucesso_teste = df_resultados_CV2_teste["leitura_sucesso"].sum()
print(f"Imagens carregadas corretement no teste: {sucesso_teste}")
print("="*50)


# Imagens carregadas incorretementeno treino
print("="*50)
falhas_treino = (df_resultados_CV2_treino["leitura_sucesso"] == False).sum()
print(f"Falhas de leitura no treinamento: {falhas_treino}")

# Imagens carregadas incorretementeno teste
falhas_teste = (df_resultados_CV2_teste["leitura_sucesso"] == False).sum()
print(f"Falhas de leitura no teste: {falhas_teste}")
print("="*50)


# calculando a taxa de sucesso - treino
print("="*50)
taxa_sucesso_treino = (sucesso_treino / len(df_resultados_CV2_treino)) * 100
print(f"Taxa de sucesso no treinamento: {taxa_sucesso_treino:.2f}%")

# teste
taxa_sucesso_teste = (sucesso_teste / len(df_resultados_CV2_teste)) * 100
print(f"Taxa de sucesso no teste: {taxa_sucesso_teste:.2f}%")
print("="*50)


# verificando falhas - treino
imagens_falhas_treino = df_resultados_CV2_treino.loc[df_resultados_CV2_treino["leitura_sucesso"] == False]
imagens_falhas_treino

# teste
imagens_falhas_teste = df_resultados_CV2_teste.loc[df_resultados_CV2_teste["leitura_sucesso"] == False]
imagens_falhas_teste

#############################################
# 1.3 - Verificação de Imagens duplicadas
#############################################

print("="*50)
print("1.3 - Verificação de Imagens duplicadas")
print("="*50)

# calculando hash
def Calcula_hash(caminho_imagem):
        imagem_binary = caminho_imagem.read_bytes()     # lê o conteúdo de um arquivo e retorna seus dados em binário (bytes).
        obj_hash = hashlib.sha256()                     # Criamos o objeto vazio.         
        obj_hash.update(imagem_binary)                  # Fornecemos os bytes por meio de update().
        hash_hexa = obj_hash.hexdigest()                # Converte o resultado do hash em uma representação hexadecimal.
        return hash_hexa


# Teste para duas imagens
# pasta = caminho_treino / "Almonds 1"
# imagem_1 = pasta / "r0_0_100.jpg"
# imagem_2 = pasta / "r0_1_100.jpg"
# hash_1 = Calcula_hash(imagem_1)
# hash_2 = Calcula_hash(imagem_2)
# print(hash_1)
# print(hash_2)


# Calculando os hashes das imagens de treinamento
hashes_treino = []
for pasta in caminho_treino.iterdir():
    if pasta.is_dir():
        for caminho_imagem in pasta.glob("*.jpg"):
            hash_imagem = Calcula_hash(caminho_imagem)
            registro = {"classe": pasta.name,
                        "imagem": caminho_imagem.name,
                        "hash": hash_imagem,
                        "conjunto": "Training"}
            hashes_treino.append(registro)



# Calculando os hashes das imagens de teste
hashes_teste = []
for pasta in caminho_teste.iterdir():
    if pasta.is_dir():
        for caminho_imagem in pasta.glob("*.jpg"):
            hash_imagem = Calcula_hash(caminho_imagem)
            registro = {"classe": pasta.name,
                        "imagem": caminho_imagem.name,
                        "hash": hash_imagem,
                        "conjunto": "Test"}
            hashes_teste.append(registro)



# Transformando os resultados em DataFrames
df_hashes_treino = pd.DataFrame(hashes_treino)
df_hashes_teste = pd.DataFrame(hashes_teste)

print("Imagens analisadas no treinamento:", len(df_hashes_treino))
print("Imagens analisadas no teste:", len(df_hashes_teste))


# Visualizando os registros
print("Visualizando os registros")
print(df_hashes_treino.head())
print(df_hashes_teste.head())

print("Verificando a estrutura")
df_hashes_treino.info()
df_hashes_teste.info()

# Verificando duplicatas - treino
duplicatas_treino = df_hashes_treino.duplicated(subset="hash")
print("Quantidade de duplicatas no treinamento:", duplicatas_treino.sum())

df_duplicatas_treino = df_hashes_treino.loc[
    df_hashes_treino.duplicated(subset="hash", keep=False)
]

print(df_duplicatas_treino)


# Verificando duplicatas - teste
duplicatas_teste = df_hashes_teste.duplicated(subset="hash")
print("Quantidade de duplicatas no teste:", duplicatas_teste.sum())

df_duplicatas_teste = df_hashes_teste.loc[
    df_hashes_teste.duplicated(subset="hash", keep=False)
]
print(df_duplicatas_teste)


# Identificando duplicatas entre treinamento e teste
hashes_unicos_treino = set(df_hashes_treino["hash"])
hashes_unicos_teste = set(df_hashes_teste["hash"])

hashes_comuns = hashes_unicos_treino.intersection(hashes_unicos_teste)
print("Quantidade de hashes presentes em ambos os conjuntos:", len(hashes_comuns))

df_duplicatas_entre_conjuntos = pd.concat(
    [df_hashes_treino, df_hashes_teste]
)

df_duplicatas_entre_conjuntos = df_duplicatas_entre_conjuntos.loc[
    df_duplicatas_entre_conjuntos["hash"].isin(hashes_comuns)
]

print(df_duplicatas_entre_conjuntos)


##########################################################
# 1.4 - Verificando possíveis problemas de rotulagem
##########################################################

print("="*50)
print("1.4 - Verificando possíveis problemas de rotulagem")
print("="*50)

# Selecionando 10 classes aleatórias 
classes_amostra = random.sample(classes_treino, 10)
print(f"Classes selecionadas:\n{classes_amostra}")

# Selecionando imagens aleatórias das classes
imagens_amostra = []
for classe in classes_amostra:
    pasta = caminho_treino / classe
    imagens = list(pasta.glob("*.jpg"))
    imagens_selecionadas = random.sample(imagens, 5)
    for imagem in imagens_selecionadas:
        imagens_amostra.append({"classe": classe,
                                "caminho": imagem})


# Visualizando as imagens selecionadas

fig, axes = plt.subplots(10, 5, figsize=(15, 25))

for i, registro in enumerate(imagens_amostra):
    imagem = cv2.imread(str(registro["caminho"]))
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    linha = i // 5
    coluna = i % 5
    axes[linha, coluna].imshow(imagem)
    axes[linha, coluna].set_title(registro["classe"], fontsize=9)
    axes[linha, coluna].axis("off")
plt.tight_layout()
plt.show()


##########################################################################
# 1.5 - Analizando o balanciamento das classes
##########################################################################

print("="*50)
print("1.5 - Analizando o balanciamento das classes")
print("="*50)


# Classes com maior quantidade de imagens no treinamento
df_treino_ordenado = df_treino.sort_values(by="quantidade", ascending=False)
print("Classes com maior quantidade de imagens no treinamento:")
print(df_treino_ordenado.head(10))

# Classes com menor quantidade de imagens no treinamento
df_treino_ordenado_crescente = df_treino.sort_values(by="quantidade", ascending=True)
print("Classes com menor quantidade de imagens no treinamento:")
print(df_treino_ordenado_crescente.head(10))

print("Estatísticas do treinamento:")
print(df_treino["quantidade"].describe())


# Classes com maior quantidade de imagens no teste
df_teste_ordenado = df_teste.sort_values(by="quantidade", ascending=False)
print("Classes com maior quantidade de imagens no teste:")
print(df_teste_ordenado.head(10))

# Classes com menor quantidade de imagens no teste
df_teste_ordenado_crescente = df_teste.sort_values(by="quantidade", ascending=True)
print("Classes com menor quantidade de imagens no teste:")
print(df_teste_ordenado_crescente.head(10))


print("Estatísticas do teste:")
print(df_teste["quantidade"].describe())


# Comparando as quantidades de imagens por classe
df_comparacao = pd.merge(df_treino, df_teste, 
                         on="classe", suffixes=("_treino", "_teste"))
print("Quantidade de imagens por classe:")
print(df_comparacao.head())


# Calculando a proporção de treinamento e teste

df_comparacao["total"] = (
    df_comparacao["quantidade_treino"] +
    df_comparacao["quantidade_teste"]
)

df_comparacao["proporcao_treino"] = (
    df_comparacao["quantidade_treino"] /
    df_comparacao["total"]
)

df_comparacao["proporcao_teste"] = (
    df_comparacao["quantidade_teste"] /
    df_comparacao["total"]
)
print("Proporção de treinamento e teste:")
print(df_comparacao.head())

# Histograma do conjunto de treinamento
plt.figure(figsize=(10, 6))

plt.hist(
    df_treino["quantidade"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribuição da quantidade de imagens por classe - Treinamento")
plt.xlabel("Quantidade de imagens por classe")
plt.ylabel("Número de classes")

plt.grid(axis="y", alpha=0.3)

plt.show()


# Histograma do conjunto de teste
plt.figure(figsize=(10, 6))

plt.hist(
    df_teste["quantidade"],
    bins=20,
    edgecolor="black"
)

plt.title("Distribuição da quantidade de imagens por classe - Teste")
plt.xlabel("Quantidade de imagens por classe")
plt.ylabel("Número de classes")

plt.grid(axis="y", alpha=0.3)

plt.show()


# Selecionando as 10 classes com maior quantidade de imagens
top_10 = df_treino_ordenado.head(10)

plt.figure(figsize=(12, 6))

plt.bar(
    top_10["classe"],
    top_10["quantidade"]
)

plt.title("10 classes com maior quantidade de imagens - Treinamento")
plt.xlabel("Classes")
plt.ylabel("Quantidade de imagens")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.show()



maior_quantidade = df_treino["quantidade"].max()
menor_quantidade = df_treino["quantidade"].min()
diferenca = maior_quantidade - menor_quantidade

print("Maior quantidade:", maior_quantidade)
print("Menor quantidade:", menor_quantidade)
print("Diferença:", diferenca)

razao = maior_quantidade / menor_quantidade
print("Razão entre a maior e a menor classe:", razao)


####################################################################
# 1.6 - Exemplos de imagens e principais dificuldades do modelo
####################################################################

print("="*50)
print("1.6 - Exemplos de imagens e principais dificuldades do modelo")
print("="*50)

# Visualizando classes relacionadas a maçãs
classes_maça = [
    classe for classe in classes_treino
    if "Apple" in classe
]
print(classes_maça)

# Visualizando classes relacionadas a peras
classes_pera = [
    classe for classe in classes_treino
    if "Pear" in classe
]
print(classes_pera)

# Visualizando classes relacionadas a cerejas
classes_cereja = [
    classe for classe in classes_treino
    if classe.startswith("Cherry")
]
print(classes_cereja)


# Visualizando diferentes classes do mesmo grupo
# Escolha manual de classes para comparação

classes_visualizacao = [
    "Apple Braeburn 1",
    "Apple Granny Smith 1",
    "Apple Golden 1",
    "Apple Red 1",
    "Apple Pink Lady 1"
]

imagens_visualizacao = []

for classe in classes_visualizacao:
    pasta = caminho_treino / classe
    imagens = list(pasta.glob("*.jpg"))
    imagem_selecionada = random.choice(imagens)
    imagens_visualizacao.append({
        "classe": classe,
        "caminho": imagem_selecionada
    })


# Selecionar uma imagem aleatória de cada classe
def selecionar_imagens(classes, caminho_base):
    imagens_selecionadas = []
    for classe in classes:
        pasta = caminho_base / classe
        imagens = list(pasta.glob("*.jpg"))
        imagem = random.choice(imagens)
        imagens_selecionadas.append({
            "classe": classe,
            "caminho": imagem
        })

    return imagens_selecionadas


# Criar os grupos de análise

classes_macas = [
    "Apple Braeburn 1",
    "Apple Granny Smith 1",
    "Apple Golden 1",
    "Apple Pink Lady 1",
    "Apple Red Delicious 1"
]

classes_peras = [
    "Pear Abate 1",
    "Pear Forelle 1",
    "Pear Kaiser 1",
    "Pear Williams 1",
    "Pear Red 1"
]

classes_cerejas = [
    "Cherry 1",
    "Cherry Rainier 1",
    "Cherry Sour 1",
    "Cherry Wax Red 1",
    "Cherry Wax Yellow 1"
]

# Selecionando as imagens 
imagens_macas = selecionar_imagens(
    classes_macas,
    caminho_treino
)

imagens_peras = selecionar_imagens(
    classes_peras,
    caminho_treino
)

imagens_cerejas = selecionar_imagens(
    classes_cerejas,
    caminho_treino
)


# Criar uma função para visualizar as imagens

def visualizar_imagens(imagens_selecionadas, titulo):
    fig, axes = plt.subplots(
        1,
        len(imagens_selecionadas),
        figsize=(15, 4)
    )
    fig.suptitle(titulo, fontsize=16)
    for i, registro in enumerate(imagens_selecionadas):
        imagem = cv2.imread(str(registro["caminho"]))
        imagem = cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )
        axes[i].imshow(imagem)
        axes[i].set_title(
            registro["classe"],
            fontsize=9
        )
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


visualizar_imagens(
    imagens_macas,
    "Comparação visual entre variedades de maçãs"
)

visualizar_imagens(
    imagens_peras,
    "Comparação visual entre variedades de peras"
)

visualizar_imagens(
    imagens_cerejas,
    "Comparação visual entre variedades de cerejas"
)

##################################################################
# 1.7 - Divisão dos dados e prevenção de vazamento de dados
##################################################################

print("="*50)
print("1.7 - Divisão dos dados e prevenção de vazamento de dados")
print("="*50)

# Verificando hashes compartilhados entre treinamento e teste
hashes_treino = set(df_hashes_treino["hash"])
hashes_teste = set(df_hashes_teste["hash"])
hashes_comuns = hashes_treino.intersection(hashes_teste)
print("Quantidade de hashes compartilhados:", len(hashes_comuns))


# Investigando nomes de arquivos relacionados
pasta = caminho_treino / "Apple Braeburn 1"
imagens = list(pasta.glob("*.jpg"))
for imagem in imagens[:20]:
    print(imagem.name)


# Criando uma estrutura com os caminhos das imagens
def criar_dataframe_imagens(caminho_base):
    registros = []
    for pasta in caminho_base.iterdir():
        if pasta.is_dir():
            for imagem in pasta.glob("*.jpg"):
                registros.append({
                    "classe": pasta.name,
                    "caminho": imagem
                })
    df = pd.DataFrame(registros)
    return df


df_imagens_treino = criar_dataframe_imagens(caminho_treino)
df_imagens_teste = criar_dataframe_imagens(caminho_teste)

print("Treinamento:")
print(df_imagens_treino.shape)

print("\nTeste:")
print(df_imagens_teste.shape)

print(df_imagens_treino.head())


# Selecionando 25% das imagens de treinamento
df_treino_reduzido = (
    df_imagens_treino
    .groupby("classe", group_keys=False)
    .sample(frac=0.25, random_state=42)
)

print("Quantidade de imagens selecionadas:", len(df_treino_reduzido))


# Selecionando 25% das imagens de teste
df_teste_reduzido = (
    df_imagens_teste
    .groupby("classe", group_keys=False)
    .sample(frac=0.25, random_state=42)
)

print("Quantidade de imagens selecionadas:", len(df_teste_reduzido))


print("===== DATASET REDUZIDO =====")

print("\nTreinamento:")
print(df_treino_reduzido.shape)

print("\nTeste:")
print(df_teste_reduzido.shape)

print("\nTotal:")
print(
    len(df_treino_reduzido) +
    len(df_teste_reduzido)
)


classes_reduzidas_treino = set(
    df_treino_reduzido["classe"]
)

classes_reduzidas_teste = set(
    df_teste_reduzido["classe"]
)

print("Classes no treinamento:", len(classes_reduzidas_treino))
print("Classes no teste:", len(classes_reduzidas_teste))

# verificamos se as classes são iguais
print(
    "Classes ausentes no treinamento:",
    set(classes_treino) - classes_reduzidas_treino
)

print(
    "Classes ausentes no teste:",
    set(classes_teste) - classes_reduzidas_teste
)


# Verificando o balanciamento após a redução
contagem_reduzida_treino = (
    df_treino_reduzido["classe"]
    .value_counts()
    .sort_index()
)

contagem_reduzida_teste = (
    df_teste_reduzido["classe"]
    .value_counts()
    .sort_index()
)

print("Estatísticas do treinamento reduzido:")
print(contagem_reduzida_treino.describe())
print("\nEstatísticas do teste reduzido:")
print(contagem_reduzida_teste.describe())

# Visualizando a distribuição após a redução 
# Treino
plt.figure(figsize=(10, 5))
plt.hist(
    contagem_reduzida_treino,
    bins=20,
    edgecolor="black"
)
plt.title("Distribuição de imagens por classe - Treinamento reduzido")
plt.xlabel("Quantidade de imagens por classe")
plt.ylabel("Número de classes")
plt.grid(axis="y", alpha=0.3)
plt.show()

# Teste
plt.figure(figsize=(10, 5))
plt.hist(
    contagem_reduzida_teste,
    bins=20,
    edgecolor="black"
)
plt.title("Distribuição de imagens por classe - Teste reduzido")
plt.xlabel("Quantidade de imagens por classe")
plt.ylabel("Número de classes")
plt.grid(axis="y", alpha=0.3)
plt.show()


# Salvando a seleção
df_treino_reduzido.to_csv(
    "treino_reduzido.csv",
    index=False
)

df_teste_reduzido.to_csv(
    "teste_reduzido.csv",
    index=False
)

####################################################################
# Etapa 2 - Pré - processamento das imagens 
####################################################################

print("="*50)
print("ETAPA 02 - Pré-processamento das imagens")
print("="*50)

df_treino = pd.read_csv("treino_reduzido.csv")
df_teste = pd.read_csv("teste_reduzido.csv")

print("Treino:", df_treino.shape, "| Teste:", df_teste.shape)
print(df_treino.head())

# Teste de sanidade: o caminho salvo no CSV existe na sua máquina?
primeiro = Path(df_treino.loc[0, "caminho"])
print(primeiro, "->", primeiro.exists())

def carregar_rgb(caminho):
    img = cv2.imread(str(caminho))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

amostra = df_treino.sample(500, random_state=42)

formatos, tipos = set(), set()
brilho_borda = []   # média dos pixels da borda da imagem (deve ser fundo)

for caminho in amostra["caminho"]:
    img = carregar_rgb(caminho)
    formatos.add(img.shape)
    tipos.add(str(img.dtype))
    borda = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    brilho_borda.append(borda.mean())

print("Formatos encontrados:", formatos)
print("Tipos de dado:", tipos)
print(f"Brilho médio da borda: {np.mean(brilho_borda):.1f} "
      f"(mín {np.min(brilho_borda):.1f}, máx {np.max(brilho_borda):.1f})")

# Visualizando como é o fundo "branco" das imagens
amostra_vis = df_treino.sample(6, random_state=1)

fig, axes = plt.subplots(3, 6, figsize=(16, 8))

for i, (classe, caminho) in enumerate(zip(amostra_vis["classe"], amostra_vis["caminho"])):
    img = carregar_rgb(caminho)
    cinza = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    axes[0, i].imshow(img)
    axes[0, i].set_title(classe, fontsize=8)

    axes[1, i].imshow(cinza > 250, cmap="gray")
    axes[1, i].set_title("pixels > 250", fontsize=8)

    axes[2, i].imshow(cinza > 235, cmap="gray")
    axes[2, i].set_title("pixels > 235", fontsize=8)

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show() 

classes_cmp = ["Apple Braeburn 1", "Apple Golden 1", "Apple Granny Smith 1", "Apple Red 1"]

fig, axes = plt.subplots(len(classes_cmp), 7, figsize=(18, 10))
titulos = ["RGB", "H (matiz)", "S (saturação)", "V (brilho)", "L (luminosidade)", "a (verde↔vermelho)", "b (azul↔amarelo)"]

for i, classe in enumerate(classes_cmp):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=3)["caminho"].iloc[0]
    rgb = carregar_rgb(caminho)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)   # H: 0-179, S e V: 0-255
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)   # L, a, b: 0-255 em uint8

    canais = [rgb, hsv[..., 0], hsv[..., 1], hsv[..., 2], lab[..., 0], lab[..., 1], lab[..., 2]]
    for j, canal in enumerate(canais):
        axes[i, j].imshow(canal, cmap=None if j == 0 else "gray")
        axes[i, j].axis("off")
        if i == 0:
            axes[i, j].set_title(titulos[j], fontsize=9)
    axes[i, 0].set_ylabel(classe)
    axes[i, 0].text(-5, 50, classe, rotation=90, va="center", ha="right", fontsize=8)

plt.tight_layout()
plt.show()

# ============================================================
# 2.4 - Redução de ruído: comparação de filtros
# Objetivo: verificar se algum filtro compensa. O risco é borrar
# a textura da casca, que será usada na Etapa 4 (LBP, GLCM).
# Filtros testados: Gaussiano, Mediana e Bilateral.
# Métrica: variância do Laplaciano (quanto MAIOR, mais nítida a
# imagem; se o filtro derruba muito esse valor, ele está borrando).
# ============================================================

# --- Dicionário de filtros (cada um recebe uma imagem RGB e devolve outra) ---
filtros = {
    "Original":      lambda im: im,
    "Gaussiano 3x3": lambda im: cv2.GaussianBlur(im, (3, 3), 0),   # média ponderada da vizinhança
    "Mediana 3":     lambda im: cv2.medianBlur(im, 3),             # mediana da vizinhança (bom contra pontos isolados)
    "Bilateral":     lambda im: cv2.bilateralFilter(im, d=5, sigmaColor=50, sigmaSpace=50),  # suaviza preservando bordas
}

def nitidez(img_rgb):
    """Variância do Laplaciano da imagem em tons de cinza.
    O Laplaciano responde a bordas e detalhes finos; sua variância
    funciona como medida simples de nitidez/textura."""
    cinza = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(cinza, cv2.CV_64F).var()

# ------------------------------------------------------------
# Parte A - Comparação visual em frutas com textura marcante
# ------------------------------------------------------------
classes_filtro = ["Apple Golden 1", "Avocado 1", "Ginger 1", "Potato Red 1"]

fig, axes = plt.subplots(len(classes_filtro), len(filtros), figsize=(14, 14))

for i, classe in enumerate(classes_filtro):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=5)["caminho"].iloc[0]
    rgb = carregar_rgb(caminho)

    for j, (nome, func) in enumerate(filtros.items()):
        resultado = func(rgb)
        # interpolation="nearest" mostra os pixels reais, sem suavização do matplotlib
        axes[i, j].imshow(resultado, interpolation="nearest")
        axes[i, j].set_title(f"{nome}\nnitidez = {nitidez(resultado):.0f}", fontsize=9)
        axes[i, j].axis("off")
    axes[i, 0].text(-5, 50, classe, rotation=90, va="center", ha="right", fontsize=8)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Parte B - Medição em 300 imagens do treino (todas as classes)
# Para cada filtro, calculamos quanto da nitidez original sobrou:
# razão = nitidez após o filtro / nitidez original (1.0 = nada mudou)
# ------------------------------------------------------------
amostra_filtros = df_treino.sample(300, random_state=7)
razoes = {nome: [] for nome in filtros if nome != "Original"}

for caminho in amostra_filtros["caminho"]:
    rgb = carregar_rgb(caminho)
    base = nitidez(rgb)
    if base == 0:      # evita divisão por zero (imagem totalmente uniforme)
        continue
    for nome, func in filtros.items():
        if nome == "Original":
            continue
        razoes[nome].append(nitidez(func(rgb)) / base)

print("Nitidez preservada após cada filtro (média das 300 imagens):")
for nome, valores in razoes.items():
    print(f"  {nome:15s} -> {np.mean(valores)*100:.1f}% da nitidez original "
          f"(mediana {np.median(valores)*100:.1f}%)")

# ============================================================
# 2.5 - Ajuste de contraste: CLAHE
# Objetivo: avaliar se o CLAHE realça a textura sem alterar a cor.
# Versões comparadas:
#   (1) Original
#   (2) CLAHE só no canal L do Lab (preserva a cor, em teoria)
#   (3) CLAHE em cada canal RGB (versão "errada", só para comparação)
# ============================================================

def clahe_lab(img_rgb, clip=2.0, grade=(4, 4)):
    """Aplica CLAHE apenas na luminosidade (canal L do Lab).
    grade=(4,4): imagem dividida em 4x4 quadrados (as imagens são pequenas, 100x100)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)                      # separa os 3 canais
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grade)
    L2 = clahe.apply(L)                           # só o L é modificado
    return cv2.cvtColor(cv2.merge([L2, a, b]), cv2.COLOR_LAB2RGB)

def clahe_rgb(img_rgb, clip=2.0, grade=(4, 4)):
    """Versão errada: CLAHE em cada canal R, G e B separadamente."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grade)
    canais = [clahe.apply(img_rgb[..., k]) for k in range(3)]
    return np.stack(canais, axis=-1)

def mascara_objeto(img_rgb):
    """Máscara aproximada da fruta: pixels que NÃO são fundo claro.
    Usa o limiar 235, que na célula 3 deu o contorno mais limpo.
    Serve só para medir dentro da fruta (a segmentação de verdade é a Etapa 3)."""
    cinza = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cinza < 235

def metricas(orig, proc):
    """Retorna (contraste, desvio_de_cor) medidos só dentro da fruta.
    contraste: desvio padrão da luminosidade L (maior = mais contraste).
    desvio_de_cor: quanto os canais a e b mudaram em relação ao original
                   (0 = cor idêntica; quanto maior, mais a cor foi alterada)."""
    mask = mascara_objeto(orig)
    if mask.sum() == 0:
        return None
    lab_o = cv2.cvtColor(orig, cv2.COLOR_RGB2LAB).astype(float)
    lab_p = cv2.cvtColor(proc, cv2.COLOR_RGB2LAB).astype(float)
    contraste = lab_p[..., 0][mask].std()
    desvio_cor = (np.abs(lab_p[..., 1] - lab_o[..., 1])[mask]
                  + np.abs(lab_p[..., 2] - lab_o[..., 2])[mask]).mean()
    return contraste, desvio_cor

versoes = {
    "Original":        lambda im: im,
    "CLAHE em L (Lab)": clahe_lab,
    "CLAHE em RGB":    clahe_rgb,
}

# ------------------------------------------------------------
# Parte A - Comparação visual (inclui uma imagem escura: Avocado)
# ------------------------------------------------------------
classes_clahe = ["Avocado 1", "Apple Golden 1", "Apple Red 1", "Potato Red 1"]

fig, axes = plt.subplots(len(classes_clahe), len(versoes), figsize=(11, 14))

for i, classe in enumerate(classes_clahe):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=5)["caminho"].iloc[0]
    rgb = carregar_rgb(caminho)

    for j, (nome, func) in enumerate(versoes.items()):
        resultado = func(rgb)
        m = metricas(rgb, resultado)
        axes[i, j].imshow(resultado, interpolation="nearest")
        axes[i, j].set_title(f"{nome}\ncontraste={m[0]:.1f} | Δcor={m[1]:.1f}", fontsize=9)
        axes[i, j].axis("off")
    axes[i, 0].text(-5, 50, classe, rotation=90, va="center", ha="right", fontsize=8)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Parte B - Medição em 300 imagens do treino
# ------------------------------------------------------------
amostra_clahe = df_treino.sample(300, random_state=7)
resultados = {nome: {"contraste": [], "cor": []} for nome in versoes}

for caminho in amostra_clahe["caminho"]:
    rgb = carregar_rgb(caminho)
    for nome, func in versoes.items():
        m = metricas(rgb, func(rgb))
        if m is not None:
            resultados[nome]["contraste"].append(m[0])
            resultados[nome]["cor"].append(m[1])

print("Média em 300 imagens (medido só dentro da fruta):")
for nome, r in resultados.items():
    print(f"  {nome:18s} contraste = {np.mean(r['contraste']):.1f} | "
          f"alteração da cor (Δcor) = {np.mean(r['cor']):.2f}")

# ============================================================
# 2.6 - Normalização dos valores de pixel
# Objetivo: converter os pixels de inteiros 0-255 (uint8) para
# decimais 0-1 (float32), escala adequada para algoritmos de ML.
# Verificamos também se a conversão não perde informação.
# ============================================================

def normalizar(img_uint8):
    """Converte uint8 (0-255) para float32 (0-1)."""
    return img_uint8.astype(np.float32) / 255.0

def desnormalizar(img_float):
    """Operação inversa: float32 (0-1) volta para uint8 (0-255)."""
    return np.round(img_float * 255.0).astype(np.uint8)

# --- Teste em 300 imagens do treino ---
amostra_norm = df_treino.sample(300, random_state=7)
minimos, maximos, dif_max = [], [], []

for caminho in amostra_norm["caminho"]:
    rgb = carregar_rgb(caminho)
    norm = normalizar(rgb)
    volta = desnormalizar(norm)

    minimos.append(norm.min())
    maximos.append(norm.max())
    # diferença máxima entre a imagem original e a que fez o "ida e volta"
    dif_max.append(np.abs(rgb.astype(int) - volta.astype(int)).max())

print("Tipo antes:", rgb.dtype, "| Tipo depois:", norm.dtype)
print(f"Menor valor encontrado: {np.min(minimos):.3f}")
print(f"Maior valor encontrado: {np.max(maximos):.3f}")
print(f"Maior diferença após normalizar e voltar: {np.max(dif_max)} (0 = sem perda)")

# --- Custo de memória: por que NÃO normalizar o dataset inteiro de uma vez ---
n_img = len(df_treino)
bytes_uint8 = n_img * 100 * 100 * 3
bytes_float32 = bytes_uint8 * 4
print(f"\nTreino inteiro em uint8:   {bytes_uint8 / 1e9:.1f} GB")
print(f"Treino inteiro em float32: {bytes_float32 / 1e9:.1f} GB")

# ============================================================
# 2.7 - Função final do pré-processamento (entrega para a Etapa 4)
# Reúne as decisões da Etapa 2 em uma única função reutilizável.
# ============================================================

def preprocessar(img_rgb, usar_clahe=False):
    """Pré-processamento da Etapa 2.
    Entrada: imagem RGB uint8 (100x100x3).
    Saída:   imagem RGB uint8; o CLAHE é opcional.
    Decisões: sem redimensionamento (imagens já 100x100) e sem filtro de ruído
    (destrói textura). A normalização para float32 [0,1] é feita por
    quem usa a imagem, com normalizar(), para economizar memória."""
    if usar_clahe:
        return clahe_lab(img_rgb)   # definida na célula 6 (CLAHE só no canal L)
    return img_rgb

# --- Teste rápido da função ---
caminho_teste = df_treino.sample(1, random_state=11)["caminho"].iloc[0]
img = carregar_rgb(caminho_teste)
img_proc = preprocessar(img, usar_clahe=True)
img_final = normalizar(img_proc)
print("Formato:", img_final.shape, "| Tipo:", img_final.dtype,
      f"| Faixa: [{img_final.min():.2f}, {img_final.max():.2f}]")

# ============================================================
# Etapa 3 - Segmentação
# ============================================================
# ============================================================
# ETAPA 3 - SEGMENTAÇÃO
# 3.0 - Preparação: bibliotecas e carregamento dos dados
# ============================================================


print("="*50)
print("ETAPA 03 - SEGMENTAÇÃO")
print("="*50)

# Mesma seleção de imagens usada nas etapas anteriores (25% por classe)
df_treino = pd.read_csv("treino_reduzido.csv")
df_teste = pd.read_csv("teste_reduzido.csv")

def carregar_rgb(caminho):
    """Lê a imagem com OpenCV e converte de BGR para RGB."""
    img = cv2.imread(str(caminho))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

print("Treino:", df_treino.shape, "| Teste:", df_teste.shape)

# ============================================================
# 3.1 - Limiarização automática (Otsu): dois candidatos
# Objetivo: comparar duas formas de separar fruta e fundo.
#   (A) Otsu em tons de cinza: fruta = pixels mais escuros que o fundo
#   (B) Otsu no canal S do HSV: fruta = pixels com mais saturação (cor)
# Os testes incluem frutas CLARAS (coco, couve-flor, cebola branca,
# repolho branco), que são o caso mais difícil para ambos os métodos.
# Usamos apenas imagens do TREINO.
# ============================================================

def mascara_otsu_cinza(img_rgb):
    """Máscara (255 = fruta, 0 = fundo) por Otsu nos tons de cinza.
    Usamos THRESH_BINARY_INV porque a fruta é MAIS ESCURA que o fundo:
    pixels abaixo do limiar viram 255 (fruta)."""
    cinza = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    limiar, mascara = cv2.threshold(cinza, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return limiar, mascara

def mascara_otsu_saturacao(img_rgb):
    """Máscara (255 = fruta, 0 = fundo) por Otsu no canal S do HSV.
    Aqui a fruta tem MAIS saturação que o fundo branco (S perto de 0),
    então usamos THRESH_BINARY (sem inverter)."""
    s = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[..., 1]
    limiar, mascara = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return limiar, mascara

classes_seg = ["Avocado 1", "Ginger 1", "Cocos 1", "Cauliflower 1", "Onion White 2", "Cabbage white 1"]

fig, axes = plt.subplots(3, len(classes_seg), figsize=(17, 9))

for j, classe in enumerate(classes_seg):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=2)["caminho"].iloc[0]
    rgb = carregar_rgb(caminho)

    lim_c, masc_c = mascara_otsu_cinza(rgb)
    lim_s, masc_s = mascara_otsu_saturacao(rgb)

    # Fração da imagem classificada como fruta (ajuda a notar máscaras absurdas)
    area_c = (masc_c > 0).mean() * 100
    area_s = (masc_s > 0).mean() * 100

    axes[0, j].imshow(rgb)
    axes[0, j].set_title(classe, fontsize=9)
    axes[1, j].imshow(masc_c, cmap="gray")
    axes[1, j].set_title(f"Otsu cinza (limiar={lim_c:.0f})\nfruta = {area_c:.0f}% da imagem", fontsize=8)
    axes[2, j].imshow(masc_s, cmap="gray")
    axes[2, j].set_title(f"Otsu canal S (limiar={lim_s:.0f})\nfruta = {area_s:.0f}% da imagem", fontsize=8)

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# 3.2 - Limpeza da máscara e teste da união dos métodos
# Pipeline de limpeza:
#   (1) Fechamento morfológico  -> tapa buracos pequenos
#   (2) Maior contorno externo, preenchido -> descarta sujeira solta
#       e tapa buracos grandes (ex.: reflexo da cebola)
# Versões comparadas: S limpa  x  união (S OR cinza) limpa
# ============================================================

def limpar_mascara(mascara, tam_kernel=5):
    """Recebe máscara 0/255 e devolve a máscara limpa (0/255)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_kernel, tam_kernel))
    fechada = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    # RETR_EXTERNAL: só contornos externos (ignora os buracos internos)
    contornos, _ = cv2.findContours(fechada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    limpa = np.zeros_like(mascara)
    if len(contornos) == 0:
        return limpa                       # nada encontrado: máscara vazia
    maior = max(contornos, key=cv2.contourArea)
    cv2.drawContours(limpa, [maior], -1, 255, thickness=cv2.FILLED)
    return limpa

def segmentar(img_rgb, metodo="S"):
    """Segmentação completa. metodo: 'S' (canal S) ou 'uniao' (S OR cinza)."""
    _, masc_s = mascara_otsu_saturacao(img_rgb)      # definida na célula 2
    if metodo == "uniao":
        _, masc_c = mascara_otsu_cinza(img_rgb)      # definida na célula 2
        masc_s = cv2.bitwise_or(masc_s, masc_c)
    return limpar_mascara(masc_s)

# ------------------------------------------------------------
# Parte A - Comparação visual (mesmas classes da célula 2)
# ------------------------------------------------------------
fig, axes = plt.subplots(4, len(classes_seg), figsize=(17, 12))

for j, classe in enumerate(classes_seg):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=2)["caminho"].iloc[0]
    rgb = carregar_rgb(caminho)

    _, s_bruta = mascara_otsu_saturacao(rgb)
    s_limpa = segmentar(rgb, "S")
    uniao_limpa = segmentar(rgb, "uniao")

    axes[0, j].imshow(rgb);                      axes[0, j].set_title(classe, fontsize=9)
    axes[1, j].imshow(s_bruta, cmap="gray");     axes[1, j].set_title(f"S bruta ({(s_bruta>0).mean()*100:.0f}%)", fontsize=8)
    axes[2, j].imshow(s_limpa, cmap="gray");     axes[2, j].set_title(f"S limpa ({(s_limpa>0).mean()*100:.0f}%)", fontsize=8)
    axes[3, j].imshow(uniao_limpa, cmap="gray"); axes[3, j].set_title(f"União limpa ({(uniao_limpa>0).mean()*100:.0f}%)", fontsize=8)

for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Parte B - Detecção de falhas em 500 imagens do treino
# Máscaras com área muito pequena (<10%) ou muito grande (>90%)
# da imagem são suspeitas. Não prova que estão certas ou erradas,
# só aponta onde olhar.
# ------------------------------------------------------------
amostra_seg = df_treino.sample(500, random_state=42).reset_index(drop=True)
areas = []
for caminho in amostra_seg["caminho"]:
    areas.append((segmentar(carregar_rgb(caminho), "S") > 0).mean() * 100)
amostra_seg["area_fruta_%"] = areas

print(f"Área média da fruta: {np.mean(areas):.1f}% | mínima: {np.min(areas):.1f}% | máxima: {np.max(areas):.1f}%")
print(f"Suspeitas (<10%): {(np.array(areas) < 10).sum()} | Suspeitas (>90%): {(np.array(areas) > 90).sum()} de {len(areas)}")

# As 6 menores áreas: vamos olhar o que aconteceu
piores = amostra_seg.nsmallest(6, "area_fruta_%")
print(piores[["classe", "area_fruta_%"]])

fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for j, (_, linha) in enumerate(piores.iterrows()):
    rgb = carregar_rgb(linha["caminho"])
    axes[0, j].imshow(rgb)
    axes[0, j].set_title(f"{linha['classe']}\n{linha['area_fruta_%']:.0f}%", fontsize=8)
    axes[1, j].imshow(segmentar(rgb, "S"), cmap="gray")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

# ============================================================
# 3.3 - Exemplos dos resultados da segmentação
# Mostra, para 8 classes SORTEADAS do treino (não escolhidas a dedo):
#   linha 1: imagem original
#   linha 2: contorno da máscara desenhado sobre a imagem
#   linha 3: fruta recortada sobre fundo branco (só os pixels da máscara)
# ============================================================

def recortar_fruta(img_rgb, mascara):
    """Mantém só os pixels da fruta e pinta o resto de branco."""
    saida = np.full_like(img_rgb, 255)
    saida[mascara > 0] = img_rgb[mascara > 0]
    return saida

def desenhar_contorno(img_rgb, mascara):
    """Desenha o contorno da máscara (em verde) sobre a imagem."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    saida = img_rgb.copy()
    cv2.drawContours(saida, contornos, -1, (0, 255, 0), 1)
    return saida

# Sorteia 8 classes diferentes e 1 imagem de cada (random_state fixo = reprodutível)
classes_sorteadas = df_treino["classe"].drop_duplicates().sample(8, random_state=21).tolist()

fig, axes = plt.subplots(3, 8, figsize=(20, 8))

for j, classe in enumerate(classes_sorteadas):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=4)["caminho"].iloc[0]
    rgb = carregar_rgb(caminho)
    masc = segmentar(rgb, "S")          # método final: canal S + limpeza

    axes[0, j].imshow(rgb)
    axes[0, j].set_title(classe, fontsize=9)
    axes[1, j].imshow(desenhar_contorno(rgb, masc), interpolation="nearest")
    axes[2, j].imshow(recortar_fruta(rgb, masc), interpolation="nearest")

axes[0, 0].set_ylabel("Original")
for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()

# ============================================================
# 3.4 - A união (S OR cinza) corrige as falhas observadas em 3.3?
# Reusa as MESMAS imagens da grade 3.3 (mesma classe, mesmo random_state)
# e compara: máscara só do canal S x união com Otsu em cinza.
# ============================================================

classes_falha = ["Apple 8", "Mushroom champignon 4", "Physalis with Husk 1"]

fig, axes = plt.subplots(3, len(classes_falha), figsize=(11, 11))

for j, classe in enumerate(classes_falha):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=4)["caminho"].iloc[0]
    rgb = carregar_rgb(caminho)

    masc_s = segmentar(rgb, "S")
    masc_u = segmentar(rgb, "uniao")

    # Pixels que a união acrescentou em relação ao canal S
    novos = cv2.bitwise_and(masc_u, cv2.bitwise_not(masc_s))

    axes[0, j].imshow(desenhar_contorno(rgb, masc_s), interpolation="nearest")
    axes[0, j].set_title(f"{classe}\nS limpa ({(masc_s>0).mean()*100:.1f}%)", fontsize=9)

    axes[1, j].imshow(desenhar_contorno(rgb, masc_u), interpolation="nearest")
    axes[1, j].set_title(f"União limpa ({(masc_u>0).mean()*100:.1f}%)", fontsize=9)

    axes[2, j].imshow(novos, cmap="gray", interpolation="nearest")
    axes[2, j].set_title(f"Pixels acrescentados pela união\n({(novos>0).sum()} pixels)", fontsize=9)

for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

##################################################################
# Etapa 4 - extração de características e no dataframe
##################################################################

# ============================================================
# ETAPA 4 - EXTRAÇÃO DE CARACTERÍSTICAS
# 4.0 - Preparação: bibliotecas, funções das etapas 2 e 3 e dados
# ============================================================

print("="*50)
print("ETAPA 04 - EXTRAÇÃO DE CARACTERÍSTICAS")
print("="*50)


# Funções prontas das Etapas 2 e 3 (arquivo preprocessamento.py)
from preprocessamento import carregar_rgb, preprocessar, segmentar, recortar_fruta, normalizar

# scikit-image: usado mais adiante para textura (LBP, GLCM) e HOG
try:
    import skimage
    print("scikit-image instalado, versão", skimage.__version__)
except ImportError:
    print("ATENÇÃO: instale com  pip install scikit-image")

# --- Carregando a seleção da Etapa 1 (25% de cada classe) ---
df_treino = pd.read_csv("treino_reduzido.csv")
df_treino["conjunto"] = "treino"
df_teste = pd.read_csv("teste_reduzido.csv")
df_teste["conjunto"] = "teste"

# Juntamos treino e teste numa tabela só. A coluna "conjunto" lembra a origem
# de cada imagem, para separar de novo na Etapa 5.
df_imagens = pd.concat([df_treino, df_teste]).reset_index(drop=True)

# --- Identificador único da imagem: conjunto/classe/nome do arquivo ---
def criar_id(linha):
    caminho = Path(linha["caminho"])
    return f"{linha['conjunto']}/{linha['classe']}/{caminho.name}"

df_imagens["id_imagem"] = df_imagens.apply(criar_id, axis=1)

print("Total de imagens:", len(df_imagens))
print(df_imagens["conjunto"].value_counts().to_dict())
print("Identificadores únicos?", df_imagens["id_imagem"].is_unique)
print(df_imagens.head())

# ============================================================
# 4.1 - Características de COR (medidas só dentro da fruta)
# Para cada imagem calculamos:
#   - média e desvio de L, a, b (Lab) e de S, V (HSV)
#   - H tratado como roda de cores: média do cosseno e do seno
#   - histograma do H em 12 faixas (fração de pixels em cada faixa de cor)
# ============================================================

def caracteristicas_cor(img_rgb, mascara):
    """Devolve um dicionário {nome_da_característica: valor}."""
    dentro = mascara > 0                          # True onde é fruta
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    carac = {}

    # Média e desvio de cada canal, só nos pixels da fruta
    canais = {"L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2],
              "S": hsv[..., 1], "V": hsv[..., 2]}
    for nome, canal in canais.items():
        valores = canal[dentro].astype(np.float32)
        carac[f"cor_{nome}_media"] = valores.mean()
        carac[f"cor_{nome}_desvio"] = valores.std()

    # H como roda de cores: no OpenCV H vai de 0 a 179, então ângulo = H * 2 graus
    h = hsv[..., 0][dentro].astype(np.float32)
    angulo = np.deg2rad(h * 2)
    carac["cor_H_cos"] = np.cos(angulo).mean()
    carac["cor_H_sen"] = np.sin(angulo).mean()

    # Histograma do H: fração de pixels em cada uma das 12 faixas de cor
    hist, _ = np.histogram(h, bins=12, range=(0, 180))
    hist = hist / max(hist.sum(), 1)              # vira fração (soma = 1)
    for i, valor in enumerate(hist):
        carac[f"cor_H_hist{i:02d}"] = valor

    return carac

# --- Teste em 4 imagens do treino (uma maçã vermelha, uma amarela, uma verde, um abacate) ---
exemplos = ["Apple Red 1", "Apple Golden 1", "Apple Granny Smith 1", "Avocado 1"]
linhas = {}

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for j, classe in enumerate(exemplos):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=3)["caminho"].iloc[0]
    img = preprocessar(carregar_rgb(caminho))
    mascara = segmentar(img, "uniao")
    linhas[classe] = caracteristicas_cor(img, mascara)

    axes[0, j].imshow(img);                axes[0, j].set_title(classe, fontsize=9)
    axes[1, j].imshow(mascara, cmap="gray"); axes[1, j].set_title("máscara", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

tabela = pd.DataFrame(linhas).T
print("Número de características de cor:", tabela.shape[1])
print(tabela[["cor_L_media", "cor_a_media", "cor_b_media", "cor_S_media", "cor_H_cos", "cor_H_sen"]].round(2))

# ============================================================
# 4.2 - Características de TEXTURA (medidas só dentro da fruta)
# LBP: descreve o padrão local da superfície (histograma de 10 padrões)
# GLCM: mede contraste, homogeneidade, energia, correlação e dissimilaridade
# Ambos usam a imagem em tons de cinza.
# ============================================================
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def caracteristicas_textura(img_rgb, mascara):
    """Devolve um dicionário {nome_da_característica: valor}."""
    cinza = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Encolhemos a máscara 2 pixels para não medir a borda da fruta
    # (na borda, a mudança fruta -> fundo pareceria "textura").
    nucleo = cv2.erode(mascara, np.ones((5, 5), np.uint8))
    dentro = nucleo > 0
    if dentro.sum() < 50:                 # fruta muito pequena: usa a máscara inteira
        dentro = mascara > 0

    carac = {}

    # --- LBP: cada pixel é comparado com os 8 vizinhos ---
    lbp = local_binary_pattern(cinza, P=8, R=1, method="uniform")   # códigos de 0 a 9
    hist, _ = np.histogram(lbp[dentro], bins=10, range=(0, 10))
    hist = hist / max(hist.sum(), 1)      # vira fração (soma = 1)
    for i, valor in enumerate(hist):
        carac[f"tex_lbp{i}"] = valor

    # --- GLCM: cinza reduzido a 16 tons (1 a 16); o fundo vira 0 e é ignorado ---
    quant = (cinza // 16).astype(np.uint8) + 1
    quant[mascara == 0] = 0
    glcm = graycomatrix(quant, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],   # 4 direções
                        levels=17, symmetric=True, normed=False)
    glcm = glcm[1:, 1:, :, :].astype(np.float64)                  # descarta o fundo
    soma = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = glcm / np.where(soma == 0, 1, soma)                     # vira proporção
    for prop in ["contrast", "homogeneity", "energy", "correlation", "dissimilarity"]:
        carac[f"tex_glcm_{prop}"] = graycoprops(glcm, prop).mean() # média das 4 direções

    return carac

# --- Teste: 4 frutas com texturas bem diferentes (lisa x rugosa) ---
exemplos_tex = ["Apple Granny Smith 1", "Avocado 1", "Ginger 1", "Cauliflower 1"]
linhas = {}

fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
for j, classe in enumerate(exemplos_tex):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=3)["caminho"].iloc[0]
    img = preprocessar(carregar_rgb(caminho))
    mascara = segmentar(img, "uniao")
    linhas[classe] = caracteristicas_textura(img, mascara)
    axes[j].imshow(img); axes[j].set_title(classe, fontsize=9); axes[j].axis("off")
plt.tight_layout(); plt.show()

tabela_tex = pd.DataFrame(linhas).T
print("Número de características de textura:", tabela_tex.shape[1])
print(tabela_tex[[c for c in tabela_tex.columns if "glcm" in c]].round(3))

# ============================================================
# 4.3 - Características de FORMA (calculadas sobre a máscara da Etapa 3)
# Área, perímetro, circularidade, proporção, preenchimento, solidez
# e os 7 momentos de Hu (formato que não muda com rotação ou tamanho).
# ============================================================

def caracteristicas_forma(mascara):
    """Devolve um dicionário {nome_da_característica: valor}.
    A máscara tem 0 (fundo) e 255 (fruta)."""
    carac = {}
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contornos) == 0:        # máscara vazia: devolve zeros
        for nome in ["area", "perimetro", "circularidade", "proporcao", "extensao", "solidez"]:
            carac[f"forma_{nome}"] = 0.0
        for i in range(7):
            carac[f"forma_hu{i + 1}"] = 0.0
        return carac

    cnt = max(contornos, key=cv2.contourArea)       # contorno da fruta
    area = cv2.contourArea(cnt)
    perimetro = cv2.arcLength(cnt, True)
    x, y, largura, altura = cv2.boundingRect(cnt)   # caixa que envolve a fruta
    area_convexa = cv2.contourArea(cv2.convexHull(cnt))

    carac["forma_area"] = area / mascara.size                 # fração da imagem (0 a 1)
    carac["forma_perimetro"] = perimetro / mascara.shape[0]   # relativo ao lado da imagem
    carac["forma_circularidade"] = 4 * np.pi * area / (perimetro ** 2) if perimetro > 0 else 0.0
    carac["forma_proporcao"] = largura / altura if altura > 0 else 0.0
    carac["forma_extensao"] = area / (largura * altura) if largura * altura > 0 else 0.0
    carac["forma_solidez"] = area / area_convexa if area_convexa > 0 else 0.0

    # Momentos de Hu: usamos o logaritmo (mantendo o sinal) porque os valores originais
    # são muito pequenos (da ordem de 0,0001) e ficariam "invisíveis" para o classificador.
    hu = cv2.HuMoments(cv2.moments(cnt)).flatten()
    for i, valor in enumerate(hu):
        carac[f"forma_hu{i + 1}"] = -np.sign(valor) * np.log10(abs(valor) + 1e-30)

    return carac

# --- Teste: formas bem diferentes (redonda, oval, alongada, irregular) ---
exemplos_forma = ["Cocos 1", "Avocado 1", "Eggplant long 1", "Pineapple Mini 1"]
linhas = {}

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for j, classe in enumerate(exemplos_forma):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=3)["caminho"].iloc[0]
    img = preprocessar(carregar_rgb(caminho))
    mascara = segmentar(img, "uniao")
    linhas[classe] = caracteristicas_forma(mascara)
    axes[0, j].imshow(img);                  axes[0, j].set_title(classe, fontsize=9)
    axes[1, j].imshow(mascara, cmap="gray"); axes[1, j].set_title("máscara", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout(); plt.show()

tabela_forma = pd.DataFrame(linhas).T
print("Número de características de forma:", tabela_forma.shape[1])
print(tabela_forma[["forma_area", "forma_circularidade", "forma_proporcao",
                      "forma_extensao", "forma_solidez"]].round(3))

# ============================================================
# 4.4 - Extração completa de UMA imagem e teste de tempo
# Junta cor + textura + forma. Antes de rodar nas 50.056 imagens,
# medimos o tempo em 200 imagens para estimar a duração total.
# ============================================================

def extrair_caracteristicas(caminho):
    """Lê uma imagem e devolve um dicionário com todas as características."""
    img = preprocessar(carregar_rgb(caminho))
    mascara = segmentar(img, "uniao")

    # Segurança: se a segmentação não achou fruta nenhuma, usamos a imagem inteira
    # e deixamos uma marca (coluna de controle, que NÃO é característica).
    vazia = int((mascara > 0).sum() == 0)
    if vazia:
        mascara = np.full_like(mascara, 255)

    carac = {}
    carac.update(caracteristicas_cor(img, mascara))
    carac.update(caracteristicas_textura(img, mascara))
    carac.update(caracteristicas_forma(mascara))
    carac["controle_mascara_vazia"] = vazia
    return carac

# --- Teste de tempo em 200 imagens ---
amostra_tempo = df_imagens.sample(200, random_state=0)

inicio = time.perf_counter()
resultados = [extrair_caracteristicas(c) for c in amostra_tempo["caminho"]]
duracao = time.perf_counter() - inicio

df_teste_tempo = pd.DataFrame(resultados)
colunas_carac = [c for c in df_teste_tempo.columns if not c.startswith("controle_")]

print(f"Tempo por imagem: {duracao / 200 * 1000:.1f} ms")
print(f"Estimativa para {len(df_imagens)} imagens: {duracao / 200 * len(df_imagens) / 60:.1f} minutos")
print("Número de características:", len(colunas_carac))
print("Valores ausentes (NaN):", int(df_teste_tempo[colunas_carac].isna().sum().sum()))
print("Valores infinitos:", int(np.isinf(df_teste_tempo[colunas_carac]).sum().sum()))
print("Máscaras vazias:", int(df_teste_tempo["controle_mascara_vazia"].sum()))

# ============================================================
# 4.5 - Extração completa (50.056 imagens) e construção do dataframe
# Processa em blocos de 5.000 imagens e salva cada bloco num arquivo.
# Se o VS Code travar ou o kernel reiniciar, basta rodar a célula de novo:
# os blocos já prontos são pulados e o trabalho continua de onde parou.
# ============================================================

TAM_BLOCO = 5000
pasta_blocos = Path("caracteristicas_parciais")
pasta_blocos.mkdir(exist_ok=True)

falhas = []                                   # imagens que não puderam ser processadas
n_blocos = int(np.ceil(len(df_imagens) / TAM_BLOCO))
inicio_total = time.perf_counter()

for b in range(n_blocos):
    arquivo = pasta_blocos / f"bloco_{b:03d}.csv"
    if arquivo.exists():
        print(f"Bloco {b + 1}/{n_blocos}: já existe, pulando")
        continue

    bloco = df_imagens.iloc[b * TAM_BLOCO:(b + 1) * TAM_BLOCO]
    registros = []
    for id_img, classe, conjunto, caminho in zip(bloco["id_imagem"], bloco["classe"],
                                                 bloco["conjunto"], bloco["caminho"]):
        try:
            registro = {"id_imagem": id_img, "classe": classe, "conjunto": conjunto}
            registro.update(extrair_caracteristicas(caminho))
            registros.append(registro)
        except Exception as erro:             # uma imagem ruim não derruba o processo
            falhas.append((id_img, str(erro)))

    pd.DataFrame(registros).to_csv(arquivo, index=False)
    minutos = (time.perf_counter() - inicio_total) / 60
    print(f"Bloco {b + 1}/{n_blocos}: {len(registros)} imagens ok | {minutos:.1f} min desde o início")

# --- Junta todos os blocos na tabela final ---
df_carac = pd.concat([pd.read_csv(a) for a in sorted(pasta_blocos.glob("bloco_*.csv"))],
                     ignore_index=True)
df_carac.to_csv("caracteristicas.csv", index=False)

print("\nTabela final:", df_carac.shape)
print("Imagens com falha:", len(falhas))
print(df_carac.head())

# ============================================================
# 4.6 - Verificação da tabela final
# ============================================================

colunas_carac = [c for c in df_carac.columns
                 if c.startswith(("cor_", "tex_", "forma_"))]

print("Linhas:", len(df_carac), "| Colunas de características:", len(colunas_carac))
print("Identificadores únicos?", df_carac["id_imagem"].is_unique)
print("Classes:", df_carac["classe"].nunique())
print("Conjuntos:", df_carac["conjunto"].value_counts().to_dict())
print("Valores ausentes (NaN):", int(df_carac[colunas_carac].isna().sum().sum()))
print("Valores infinitos:", int(np.isinf(df_carac[colunas_carac]).sum().sum()))
print("Máscaras vazias:", int(df_carac["controle_mascara_vazia"].sum()))

# --- As três configurações a comparar na Etapa 5 ---
cols_cor = [c for c in colunas_carac if c.startswith("cor_")]
cols_tex = [c for c in colunas_carac if c.startswith("tex_")]
cols_forma = [c for c in colunas_carac if c.startswith("forma_")]
configuracoes = {
    "A: so cor": cols_cor,
    "B: cor + textura": cols_cor + cols_tex,
    "C: cor + textura + forma": cols_cor + cols_tex + cols_forma,
}
for nome, colunas in configuracoes.items():
    print(f"{nome}: {len(colunas)} características")

# --- Estatísticas de algumas características (conferir se são razoáveis) ---
print(df_carac[["cor_L_media", "cor_S_media", "tex_glcm_contrast",
                  "forma_area", "forma_circularidade"]].describe().round(3))

# ============================================================
# 4.7 - Correção de classes escritas de formas diferentes
# O dataset tem uma classe cujo nome muda só nas maiúsculas/minúsculas entre
# as pastas Training e Test (identificado na Etapa 1). Aqui unificamos o nome,
# usando a grafia do conjunto de treino como padrão.
# ============================================================

# Passo 1: encontrar classes com mais de uma grafia
grupos = df_carac.groupby(df_carac["classe"].str.casefold())["classe"].unique()
duplicadas = grupos[grupos.apply(len) > 1]
print("Classes com mais de uma grafia:", len(duplicadas))
for chave, grafias in duplicadas.items():
    print("\n", list(grafias))
    print(df_carac[df_carac["classe"].isin(grafias)].groupby(["classe", "conjunto"]).size())

# Passo 2: unificar usando a grafia do treino
mapa = {}
for chave, grafias in duplicadas.items():
    grafias_treino = df_carac.loc[(df_carac["conjunto"] == "treino")
                                  & df_carac["classe"].isin(grafias), "classe"].unique()
    padrao = grafias_treino[0] if len(grafias_treino) > 0 else grafias[0]
    for g in grafias:
        mapa[g] = padrao
print("\nCorreção aplicada:", mapa)
df_carac["classe"] = df_carac["classe"].replace(mapa)

# Passo 3: conferir
print("\nClasses depois da correção:", df_carac["classe"].nunique())
classes_treino_c = set(df_carac.loc[df_carac["conjunto"] == "treino", "classe"])
classes_teste_c = set(df_carac.loc[df_carac["conjunto"] == "teste", "classe"])
print("Mesmas classes no treino e no teste?", classes_treino_c == classes_teste_c)

# Passo 4: salvar de novo a tabela corrigida
df_carac.to_csv("caracteristicas.csv", index=False)

# ============================================================
# 4.8 - HOG compacto (Histograma de Gradientes Orientados)
# O HOG divide a imagem em quadrados de 25x25 pixels e, em cada um, conta para
# que direção apontam as bordas (9 direções). Os quadrados são agrupados em
# blocos de 2x2 e normalizados. Resultado: 324 números por imagem.
# Calculado em tons de cinza, sobre a fruta recortada (fundo branco).
# ============================================================
from skimage.feature import hog
from skimage import exposure

def caracteristicas_hog(img_rgb, mascara):
    """Devolve um vetor com 324 números (float32)."""
    fruta = recortar_fruta(img_rgb, mascara)          # fundo branco constante
    cinza = cv2.cvtColor(fruta, cv2.COLOR_RGB2GRAY)
    return hog(cinza, orientations=9, pixels_per_cell=(25, 25), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)

def extrair_hog(caminho):
    """Lê uma imagem, segmenta e devolve o HOG dela."""
    img = preprocessar(carregar_rgb(caminho))
    mascara = segmentar(img, "uniao")
    if (mascara > 0).sum() == 0:                      # segurança: máscara vazia
        mascara = np.full_like(mascara, 255)
    return caracteristicas_hog(img, mascara)

# --- Visualização: como o HOG "enxerga" 4 frutas ---
exemplos_hog = ["Apple Granny Smith 1", "Pineapple Mini 1", "Eggplant long 1", "Cocos 1"]
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for j, classe in enumerate(exemplos_hog):
    caminho = df_treino[df_treino["classe"] == classe].sample(1, random_state=3)["caminho"].iloc[0]
    img = preprocessar(carregar_rgb(caminho))
    mascara = segmentar(img, "uniao")
    cinza = cv2.cvtColor(recortar_fruta(img, mascara), cv2.COLOR_RGB2GRAY)
    _, imagem_hog = hog(cinza, orientations=9, pixels_per_cell=(25, 25), cells_per_block=(2, 2),
                        block_norm="L2-Hys", visualize=True)
    axes[0, j].imshow(img); axes[0, j].set_title(classe, fontsize=9)
    axes[1, j].imshow(exposure.rescale_intensity(imagem_hog, in_range=(0, 10)), cmap="gray")
    axes[1, j].set_title("HOG", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout(); plt.show()

# --- Teste de tempo em 200 imagens ---
amostra_hog = df_imagens.sample(200, random_state=0)
inicio = time.perf_counter()
teste_hog = np.vstack([extrair_hog(c) for c in amostra_hog["caminho"]])
duracao = time.perf_counter() - inicio
print(f"Formato: {teste_hog.shape} | tipo: {teste_hog.dtype}")
print(f"Tempo por imagem: {duracao / 200 * 1000:.1f} ms")
print(f"Estimativa para {len(df_imagens)} imagens: {duracao / 200 * len(df_imagens) / 60:.1f} minutos")
print("NaN:", int(np.isnan(teste_hog).sum()), "| infinitos:", int(np.isinf(teste_hog).sum()))

# ============================================================
# 4.9 - Extração completa do HOG (50.056 imagens), em blocos de 5.000
# Cada bloco é salvo; se travar, rode a célula de novo e ela continua.
# Resultado final: hog.npz, com os identificadores (ids) e a matriz HOG (X).
# ============================================================

TAM_BLOCO = 5000
pasta_hog = Path("hog_parciais")
pasta_hog.mkdir(exist_ok=True)

falhas_hog = []
n_blocos = int(np.ceil(len(df_imagens) / TAM_BLOCO))
inicio_total = time.perf_counter()

for b in range(n_blocos):
    arquivo = pasta_hog / f"bloco_{b:03d}.npz"
    if arquivo.exists():
        print(f"Bloco {b + 1}/{n_blocos}: já existe, pulando")
        continue
    bloco = df_imagens.iloc[b * TAM_BLOCO:(b + 1) * TAM_BLOCO]
    ids, linhas = [], []
    for id_img, caminho in zip(bloco["id_imagem"], bloco["caminho"]):
        try:
            linhas.append(extrair_hog(caminho))
            ids.append(id_img)
        except Exception as erro:
            falhas_hog.append((id_img, str(erro)))
    np.savez(arquivo, ids=np.array(ids), X=np.vstack(linhas))
    minutos = (time.perf_counter() - inicio_total) / 60
    print(f"Bloco {b + 1}/{n_blocos}: {len(ids)} imagens ok | {minutos:.1f} min desde o início")

# --- Junta os blocos e salva ---
ids_todos, X_todos = [], []
for a in sorted(pasta_hog.glob("bloco_*.npz")):
    d = np.load(a)
    ids_todos.append(d["ids"]); X_todos.append(d["X"])
ids_todos = np.concatenate(ids_todos)
X_todos = np.vstack(X_todos)
np.savez("hog.npz", ids=ids_todos, X=X_todos)

print("\nHOG:", X_todos.shape, "| falhas:", len(falhas_hog))
print("Mesmos identificadores da tabela principal?", set(ids_todos) == set(df_carac["id_imagem"]))
print("NaN:", int(np.isnan(X_todos).sum()), "| infinitos:", int(np.isinf(X_todos).sum()))



# ============================================================
# ETAPA 5 - CLASSIFICAÇÃO E AVALIAÇÃO DOS MODELOS
# ============================================================

print("="*50)
print("ETAPA 05 - CLASSIFICAÇÃO E AVALIAÇÃO DOS MODELOS")
print("="*50)


# Preparação e carregamento dos dados
import seaborn as sns
import time
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report)

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# 1 -  CARREGAMENTO DO DATAFRAME DE CARACTERÍSTICAS
# ------------------------------------------------------------

df = pd.read_csv("caracteristicas.csv")

print("Dimensão do dataframe:", df.shape)
print(df.head())

# ------------------------------------------------------------
# 2 - CARREGAMENTO DAS CARACTERÍSTICAS HOG
# ------------------------------------------------------------

# Carregar arquivo HOG
dados_hog = np.load("hog.npz")

# Recuperar matriz e identificadores
matriz_hog = dados_hog["X"]
ids_hog = dados_hog["ids"]

print("Dimensão da matriz HOG:", matriz_hog.shape)
print("Quantidade de identificadores:", len(ids_hog))

# ------------------------------------------------------------
# 3 - CONFERÊNCIA DOS IDENTIFICADORES
# ------------------------------------------------------------

print("\nIdentificadores únicos no dataframe:",
      df["id_imagem"].is_unique)

print("Identificadores únicos no HOG:",
      len(np.unique(ids_hog)) == len(ids_hog))

print("Mesmos identificadores nos dois arquivos:",
      set(df["id_imagem"]) == set(ids_hog))

# ------------------------------------------------------------
# 4 - VERIFICAÇÃO DA INTEGRIDADE DOS DADOS
# ------------------------------------------------------------

print("\nValores ausentes:", df.isna().sum().sum())

print("Valores infinitos:",
      np.isinf(df.select_dtypes(include=np.number)).sum().sum())

print("\nDistribuição dos conjuntos:")
print(df["conjunto"].value_counts())

print("\nQuantidade de classes:",
      df["classe"].nunique())

# ==========================================
# RECONSTRUÇÃO DO DATAFRAME COMPLETO
# ==========================================

# Criar DataFrame das características HOG
df_hog = pd.DataFrame(
    matriz_hog,
    columns=[f"hog_{i}" for i in range(matriz_hog.shape[1])]
)

# Adicionar identificadores
df_hog.insert(0, "id_imagem", ids_hog)

# Integrar com as características tradicionais
df_completo = df.merge(
    df_hog,
    on="id_imagem",
    how="inner"
)

print("DataFrame completo:", df_completo.shape)


# ==========================================
# CONSTRUÇÃO DAS CONFIGURAÇÕES
# ==========================================

# Identificação das características por grupo
colunas_cor = [
    c for c in df.columns if c.startswith("cor_")
]

colunas_textura = [
    c for c in df.columns if c.startswith("tex_")
]

colunas_forma = [
    c for c in df.columns if c.startswith("forma_")
]

colunas_hog = [
    c for c in df_completo.columns if c.startswith("hog_")
]

# Construção das quatro configurações
configuracoes = {
    "A - Cor": colunas_cor,

    "B - Cor + Textura":
        colunas_cor + colunas_textura,

    "C - Cor + Textura + Forma":
        colunas_cor + colunas_textura + colunas_forma,

    "D - Cor + Textura + Forma + HOG":
        colunas_cor + colunas_textura + colunas_forma + colunas_hog
}

# Conferência das quantidades
print("Quantidade de características por grupo:")
print("Cor:", len(colunas_cor))
print("Textura:", len(colunas_textura))
print("Forma:", len(colunas_forma))
print("HOG:", len(colunas_hog))

print("\nConfigurações:")
for nome, colunas in configuracoes.items():
    print(f"{nome}: {len(colunas)} características")

# =================================================
#  Separação dos dados e construção do baseline
#  SEPARAÇÃO DOS DADOS
# ================================================

# Identificação dos conjuntos
dados_treino = df_completo[
    df_completo["conjunto"] == "treino"
].copy()

dados_teste = df_completo[
    df_completo["conjunto"] == "teste"
].copy()

# Variável alvo
y_treino = dados_treino["classe"]
y_teste = dados_teste["classe"]

# Matrizes de características para cada configuração
X_treino = {}
X_teste = {}

for nome, colunas in configuracoes.items():

    X_treino[nome] = dados_treino[colunas]
    X_teste[nome] = dados_teste[colunas]

# Conferência
print("Quantidade de imagens:")
print("Treino:", len(dados_treino))
print("Teste:", len(dados_teste))

print("\nQuantidade de classes:")
print("Treino:", y_treino.nunique())
print("Teste:", y_teste.nunique())

print("\nDimensões das configurações:")

for nome in configuracoes:
    print(
        f"{nome}: "
        f"Treino {X_treino[nome].shape} | "
        f"Teste {X_teste[nome].shape}"
    )

# Foi necessário a instalação da biblioteca scikit-learn (sklearn)
# %pip install scikit-learn


# ==========================================
# CONSTRUÇÃO MODELO BASELINE
# ==========================================

# Criar e treinar o modelo baseline
modelo_baseline = DummyClassifier(
    strategy="most_frequent"
)

modelo_baseline.fit(
    dados_treino[colunas_cor],
    y_treino
)

# Previsões no conjunto de treino
pred_treino_baseline = modelo_baseline.predict(
    dados_treino[colunas_cor]
)

# Avaliação no conjunto de teste
pred_teste_baseline = modelo_baseline.predict(
    dados_teste[colunas_cor]
)

# Métricas
acuracia_treino = accuracy_score(
    y_treino,
    pred_treino_baseline
)

acuracia_teste = accuracy_score(
    y_teste,
    pred_teste_baseline
)

print("RESULTADOS DO BASELINE")
print("----------------------")
print(f"Acurácia no treino: {acuracia_treino:.4f}")
print(f"Acurácia no teste:  {acuracia_teste:.4f}")

print("\nClasse prevista pelo baseline:")
print(modelo_baseline.classes_[0])


# ==========================================
# VALIDAÇÃO CRUZADA E PIPELINES
# ==========================================

# Configuração da validação cruzada
cv = StratifiedKFold( n_splits=5, shuffle=True, random_state=42)

print("Validação cruzada configurada!")
print("Número de folds:", cv.get_n_splits())
print("Estratégia: Stratified K-Fold")


# =======================================================
# CLASSIFICADOR KNN 
# Treinamento e avaliação nas quatro configurações
# =======================================================

# Avaliar todas as configurações de características
resultados_knn = {}

for configuracao, X_atual in X_treino.items():

    print("\n" + "=" * 60)
    print(f"RESULTADOS KNN - CONFIGURAÇÃO {configuracao}")
    print("=" * 60)

    # Pipeline: padronização + KNN
    pipeline_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=5,
            metric="euclidean"
        ))
    ])

    # Validação cruzada
    resultados = cross_validate(
        pipeline_knn,
        X_atual,
        y_treino,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro"
        },
        n_jobs=-1,
        return_train_score=False
    )

    resultados_knn[configuracao] = resultados

    # Resultados por fold
    print("\nRESULTADOS POR FOLD")

    for i in range(5):
        print(f"\nFold {i+1}:")
        print(f"Acurácia: {resultados['test_accuracy'][i]:.4f}")
        print(f"Precisão: {resultados['test_precision'][i]:.4f}")
        print(f"Recall:   {resultados['test_recall'][i]:.4f}")
        print(f"F1-score: {resultados['test_f1'][i]:.4f}")

    # Médias
    print("\nMÉDIAS DA VALIDAÇÃO CRUZADA")
    print("--------------------------")

    for metrica in ["accuracy", "precision", "recall", "f1"]:
        valores = resultados[f"test_{metrica}"]

        print(
            f"{metrica}: "
            f"{np.mean(valores):.4f} "
            f"+/- {np.std(valores):.4f}"
        )

# ======================================================
# CLASSIFICADOR SVM
# Treinamento e avaliação nas quatro configurações
# ======================================================

# Avaliar todas as configurações de características
resultados_svm = {}

for configuracao, X_atual in X_treino.items():

    print("\n" + "=" * 60)
    print(f"RESULTADOS SVM - CONFIGURAÇÃO {configuracao}")
    print("=" * 60)

    # Pipeline: padronização + SVM
    pipeline_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale"
        ))
    ])

    # Validação cruzada
    resultados = cross_validate(
        pipeline_svm,
        X_atual,
        y_treino,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro"
        },
        n_jobs=-1,
        return_train_score=False
    )

    resultados_svm[configuracao] = resultados

    # Resultados por fold
    print("\nRESULTADOS POR FOLD")

    for i in range(5):
        print(f"\nFold {i+1}:")
        print(f"Acurácia: {resultados['test_accuracy'][i]:.4f}")
        print(f"Precisão: {resultados['test_precision'][i]:.4f}")
        print(f"Recall:   {resultados['test_recall'][i]:.4f}")
        print(f"F1-score: {resultados['test_f1'][i]:.4f}")

    # Médias
    print("\nMÉDIAS DA VALIDAÇÃO CRUZADA")
    print("--------------------------")

    for metrica in ["accuracy", "precision", "recall", "f1"]:
        valores = resultados[f"test_{metrica}"]

        print(
            f"{metrica}: "
            f"{np.mean(valores):.4f} "
            f"+/- {np.std(valores):.4f}"
        )

# ======================================================
# CLASSIFICADOR RANDOM FOREST
# Treinamento e avaliação nas quatro configurações
# ======================================================

# Avaliar todas as configurações de características
resultados_rf = {}

for configuracao, X_atual in X_treino.items():

    print("\n" + "=" * 60)
    print(f"RESULTADOS RANDOM FOREST - CONFIGURAÇÃO {configuracao}")
    print("=" * 60)

    # Classificador Random Forest
    modelo_rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Validação cruzada
    resultados = cross_validate(
        modelo_rf,
        X_atual,
        y_treino,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro"
        },
        n_jobs=-1,
        return_train_score=False
    )

    resultados_rf[configuracao] = resultados

    # Resultados por fold
    print("\nRESULTADOS POR FOLD")

    for i in range(5):
        print(f"\nFold {i+1}:")
        print(f"Acurácia: {resultados['test_accuracy'][i]:.4f}")
        print(f"Precisão: {resultados['test_precision'][i]:.4f}")
        print(f"Recall:   {resultados['test_recall'][i]:.4f}")
        print(f"F1-score: {resultados['test_f1'][i]:.4f}")

    # Médias
    print("\nMÉDIAS DA VALIDAÇÃO CRUZADA")
    print("--------------------------")

    for metrica in ["accuracy", "precision", "recall", "f1"]:
        valores = resultados[f"test_{metrica}"]

        print(
            f"{metrica}: "
            f"{np.mean(valores):.4f} "
            f"+/- {np.std(valores):.4f}"
        )

# ===============================================
# COMPARAÇÃO INICIAL DOS CLASSIFICADORES
# Comparação entre KNN, SVM e Random Forest
# ===============================================

configuracoes = [
    "A - Cor",
    "B - Cor + Textura",
    "C - Cor + Textura + Forma",
    "D - Cor + Textura + Forma + HOG"
]

resultados = []

for configuracao in configuracoes:

    resultados.append({
        "Configuração": configuracao,

        # KNN
        "KNN - Acurácia": np.mean(resultados_knn[configuracao]["test_accuracy"]),
        "KNN - Precisão": np.mean(resultados_knn[configuracao]["test_precision"]),
        "KNN - Recall": np.mean(resultados_knn[configuracao]["test_recall"]),
        "KNN - F1-score": np.mean(resultados_knn[configuracao]["test_f1"]),

        # SVM
        "SVM - Acurácia": np.mean(resultados_svm[configuracao]["test_accuracy"]),
        "SVM - Precisão": np.mean(resultados_svm[configuracao]["test_precision"]),
        "SVM - Recall": np.mean(resultados_svm[configuracao]["test_recall"]),
        "SVM - F1-score": np.mean(resultados_svm[configuracao]["test_f1"]),

        # Random Forest
        "RF - Acurácia": np.mean(resultados_rf[configuracao]["test_accuracy"]),
        "RF - Precisão": np.mean(resultados_rf[configuracao]["test_precision"]),
        "RF - Recall": np.mean(resultados_rf[configuracao]["test_recall"]),
        "RF - F1-score": np.mean(resultados_rf[configuracao]["test_f1"])
    })

df_comparacao = pd.DataFrame(resultados)

print(df_comparacao.round(4))


# ==========================================
# EXIBIÇÃO DAS TABELAS POR MÉTRICA
# ==========================================

metricas = ["Acurácia", "Precisão", "Recall", "F1-score"]

for metrica in metricas:

    colunas = [
        "Configuração",
        f"KNN - {metrica}",
        f"SVM - {metrica}",
        f"RF - {metrica}"
    ]

    print(f"\nCOMPARAÇÃO - {metrica.upper()}")
    print("-" * 70)

    tabela = df_comparacao[colunas].copy()

    # Converter valores para percentual
    for coluna in colunas[1:]:
        tabela[coluna] = tabela[coluna].apply(
            lambda valor: f"{valor:.2%}"
        )

    print(tabela)

#==================================================
# Otimização dos hiperparâmetros — GridSearchCV
# Redução das Amostras em 30% para Treinamento
#==================================================
# Índices da amostra de treinamento
indices_treino, indices_descartados = train_test_split(
    np.arange(len(y_treino)),
    train_size=0.30,
    stratify=y_treino,
    random_state=42
)

# Criar amostras reduzidas de todas as configurações
X_treino_amostra = {
    nome: dados.iloc[indices_treino].copy()
    for nome, dados in X_treino.items()
}

y_treino_amostra = y_treino.iloc[indices_treino].copy()

# Conferir dimensões
print("Amostras de treinamento:", len(y_treino_amostra))
print("Número de classes:", y_treino_amostra.nunique())

for nome, dados in X_treino_amostra.items():
    print(nome, dados.shape)

#=================================
# Otimização do KNN
#=================================
#   Construção do Pipeline
pipeline_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

#   Definir a grade de hiperparâmetros
param_grid_knn = {
    "knn__n_neighbors": [3, 5, 7, 9, 11],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan"]
}

#   Configurar o GridSearchCV
grid_knn = GridSearchCV(
    estimator=pipeline_knn,
    param_grid=param_grid_knn,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

#   Executar a otimização do KNN
grid_knn.fit(
    X_treino_amostra["B - Cor + Textura"],
    y_treino_amostra
)


#   Exibir os resultados
print("Melhores parâmetros:")
print(grid_knn.best_params_)

print("\nMelhor F1-score macro:")
print(f"{grid_knn.best_score_:.6f}")

#=================================
# Otimização do SVM
#=================================
#Montar Pipeline do SVM
pipeline_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf"))
])

#   Definir os hiperparâmetros
param_grid_svm = {
    "svm__C": [1, 10, 100],
    "svm__gamma": ["scale", 0.01]
}

#   Configurar o GridSearchCV
grid_svm = GridSearchCV(
    estimator=pipeline_svm,
    param_grid=param_grid_svm,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

#   Executar a otimização
grid_svm.fit(
    X_treino_amostra["D - Cor + Textura + Forma + HOG"],
    y_treino_amostra
)

#   Exibir os resultados
print("Melhores parâmetros:")
print(grid_svm.best_params_)

print("\nMelhor F1-score macro:")
print(f"{grid_svm.best_score_:.6f}")

#======================================
# Otimização do Random Forest
#======================================
# Modelo base
modelo_rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

# Grade reduzida de hiperparâmetros
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [1, 2]
}

#=======================================
# Otimizar a configuração C
#=======================================

inicio = time.time()

grid_rf_c = GridSearchCV(
    estimator=modelo_rf,
    param_grid=param_grid_rf,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True
)

grid_rf_c.fit(
    X_treino_amostra["C - Cor + Textura + Forma"],
    y_treino_amostra
)

tempo_rf_c = time.time() - inicio

print("CONFIGURAÇÃO C - COR + TEXTURA + FORMA")
print("Melhores parâmetros:")
print(grid_rf_c.best_params_)

print("\nMelhor F1-score macro:")
print(f"{grid_rf_c.best_score_:.6f}")

print(f"\nTempo de execução: {tempo_rf_c:.2f} segundos")

#=================================
# Organizar os resultados
#=================================

resultados_modelos = pd.DataFrame([
    {
        "Classificador": "KNN",
        "Configuração": "B - Cor + Textura",
        "F1-score macro": grid_knn.best_score_,
        "Melhores parâmetros": grid_knn.best_params_
    },
    {
        "Classificador": "SVM",
        "Configuração": "D - Cor + Textura + Forma + HOG",
        "F1-score macro": grid_svm.best_score_,
        "Melhores parâmetros": grid_svm.best_params_
    },
    {
        "Classificador": "Random Forest",
        "Configuração": "C - Cor + Textura + Forma",
        "F1-score macro": grid_rf_c.best_score_,
        "Melhores parâmetros": grid_rf_c.best_params_
    }
])

resultados_modelos = resultados_modelos.sort_values(
    by="F1-score macro",
    ascending=False
).reset_index(drop=True)

resultados_modelos["F1-score macro (%)"] = (
    resultados_modelos["F1-score macro"] * 100
)

print(resultados_modelos)

#========================================================
# Organizar dos resultados comparação em gráfico
#========================================================
# Dados da comparação
nomes = [
    "Random Forest\n(C - Cor + Textura + Forma)",
    "KNN\n(B - Cor + Textura)",
    "SVM\n(D - Cor + Textura + Forma + HOG)"]

valores = [
    grid_rf_c.best_score_ * 100,
    grid_knn.best_score_ * 100,
    grid_svm.best_score_ * 100]

# Criar figura
fig, ax = plt.subplots(figsize=(12, 6))

# Barras horizontais
barras = ax.barh(nomes, valores, height=0.55)

# Inverter para deixar o maior resultado no topo
ax.invert_yaxis()

# Escala do eixo X
ax.set_xlim(90, 100)

# Títulos e rótulos
ax.set_title(
    "Comparação dos Classificadores Otimizados", fontsize=16, fontweight="bold", pad=20)

ax.set_xlabel( "F1-score macro (%)", fontsize=12)
ax.set_ylabel("Classificadores e configurações", fontsize=12)

# Adicionar percentual em cada barra
for barra, valor in zip(barras, valores):
    ax.text(
        valor + 0.08,
        barra.get_y() + barra.get_height() / 2,
        f"{valor:.2f}%",
        va="center",
        fontsize=12,
        fontweight="bold")

# Grade discreta
ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.35)

ax.set_axisbelow(True)

# Ajustes visuais
ax.tick_params(axis="both", labelsize=10)
plt.tight_layout()

#**********************************************
#   Salva o gráfico em uma resolução melhor
#********************************************** 
#plt.savefig(
#    "comparacao_classificadores.png",
#    dpi=300,
#    bbox_inches="tight")

plt.show()

#=============================================
# Treinamento dos modelos finais  
#==============================================
# Recuperar os melhores hiperparâmetros
print("KNN:", grid_knn.best_params_)
print("SVM:", grid_svm.best_params_)
print("Random Forest:", grid_rf_c.best_params_)

#====================================
#  Construir os modelos finais
#====================================
# KNN
modelo_knn_final = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(
        **{
            chave.replace("knn__", ""): valor
            for chave, valor in grid_knn.best_params_.items()
        }
    ))
])

# SVM
modelo_svm_final = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        **{
            chave.replace("svm__", ""): valor
            for chave, valor in grid_svm.best_params_.items()
        }
    ))
])

# Random Forest
modelo_rf_final = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    **grid_rf_c.best_params_
)

#================================================
# Treinando KNN - Configuração B
#================================================
inicio = time.time()

modelo_knn_final.fit(
    X_treino["B - Cor + Textura"],
    y_treino
)
print(f"Tempo KNN: {(time.time() - inicio)/60:.2f} minutos")

#================================================
# Treinando SVM - Configuração D
#================================================
inicio = time.time()

modelo_svm_final.fit(
    X_treino["D - Cor + Textura + Forma + HOG"],
    y_treino
)
print(f"Tempo SVM: {(time.time() - inicio)/60:.2f} minutos")

#================================================
#  Treinando Random Forest - Configuração C
#================================================
inicio = time.time() 

modelo_rf_final.fit(
    X_treino["C - Cor + Textura + Forma"],
    y_treino
)
print(f"Tempo Random Forest: {(time.time() - inicio)/60:.2f} minutos")

#===========================================
#  Avaliação final dos classificadores
#  Avaliação das métricas gerais
#===========================================

# Fazer previsões
y_pred_knn = modelo_knn_final.predict(
    X_teste["B - Cor + Textura"]
)

y_pred_svm = modelo_svm_final.predict(
    X_teste["D - Cor + Textura + Forma + HOG"]
)

y_pred_rf = modelo_rf_final.predict(
    X_teste["C - Cor + Textura + Forma"]
)

# Organizar previsões
previsoes = {
    "KNN": y_pred_knn,
    "SVM": y_pred_svm,
    "Random Forest": y_pred_rf
}

# Calcular métricas
resultados_teste = []

for nome, y_pred in previsoes.items():

    resultados_teste.append({
        "Classificador": nome,
        "Acurácia": accuracy_score(y_teste, y_pred),
        "Precisão macro": precision_score(
            y_teste, y_pred, average="macro", zero_division=0
        ),
        "Recall macro": recall_score(
            y_teste, y_pred, average="macro", zero_division=0
        ),
        "F1-score macro": f1_score(
            y_teste, y_pred, average="macro", zero_division=0
        )
    })

df_resultados_teste = pd.DataFrame(resultados_teste)

print(
    df_resultados_teste.round(4)
)


# ============================================================
# MÉTRICAS DE DESEMPENHO POR CLASSE
# ============================================================

# Organizar as previsões dos classificadores
previsoes = {
    "KNN": y_pred_knn,
    "SVM": y_pred_svm,
    "Random Forest": y_pred_rf
}

# Identificar as classes presentes no conjunto de teste
classes = sorted(y_teste.unique())

# Armazenar os relatórios individuais
relatorios_por_classe = {}

for nome, y_pred in previsoes.items():

    # Gerar relatório de classificação
    relatorio = classification_report(
        y_teste,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0
    )

    # Transformar o relatório em DataFrame
    df_relatorio = pd.DataFrame(relatorio).T

    # Manter somente as métricas das classes
    df_relatorio = df_relatorio.loc[
        [str(classe) for classe in classes],
        ["precision", "recall", "f1-score", "support"]
    ]

    # Renomear as colunas
    df_relatorio.columns = [
        "Precisão",
        "Recall",
        "F1-score",
        "Suporte"
    ]

    # Armazenar resultado
    relatorios_por_classe[nome] = df_relatorio

    print(f"\n{'='*60}")
    print(f"MÉTRICAS POR CLASSE - {nome}")
    print(f"{'='*60}")

    print(df_relatorio.round(4))


# Salvar os relatórios individuais
for nome, df_relatorio in relatorios_por_classe.items():

    arquivo = (
        "metricas_por_classe_"
        + nome.lower().replace(" ", "_")
        + ".csv"
    )

    df_relatorio.to_csv(
        arquivo,
        encoding="utf-8-sig"
    )

    print(f"Arquivo salvo: {arquivo}")


# =====================================================
# MATRIZES DE CONFUSÃO REDUZIDAS — TOP 5 CLASSES COM ERROS
# =====================================================

def matriz_confusao_top5(y_real, y_pred, nome_modelo):

    # Converter para arrays
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    # Obter todas as classes
    classes = sorted(np.unique(y_real))

    # Construir a matriz de confusão completa
    matriz = confusion_matrix(
        y_real,
        y_pred,
        labels=classes
    )

    # Calcular a quantidade de erros por classe
    erros = matriz.sum(axis=1) - matriz.diagonal()

    # Identificar as cinco classes com mais erros
    indices_top5 = np.argsort(erros)[::-1][:5]

    classes_top5 = [classes[i] for i in indices_top5]

    # Criar matriz reduzida
    matriz_reduzida = []

    for i in indices_top5:

        linha = []

        # Contagem das previsões para as cinco classes selecionadas
        for j in indices_top5:
            linha.append(matriz[i, j])

        # Agrupar previsões destinadas às demais classes
        outras = matriz[i].sum() - sum(linha)

        linha.append(outras)

        matriz_reduzida.append(linha)

    # Organizar em DataFrame
    colunas = classes_top5 + ["Outras classes"]

    df_matriz = pd.DataFrame(
        matriz_reduzida,
        index=classes_top5,
        columns=colunas
    )

    # Exibir matriz de confusão
    plt.figure(figsize=(11, 6))

    sns.heatmap(
        df_matriz,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title(
        f"Matriz de confusão reduzida — {nome_modelo}"
    )

    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")

    plt.tight_layout()
    plt.show()

    return df_matriz


# =====================================================
# EXECUÇÃO DAS MATRIZES DE CONFUSÃO REDUZIDAS
# =====================================================

matrizes_reduzidas = {}

for nome, y_pred in previsoes.items():

    print(f"\nMatriz de confusão — {nome}")

    df_matriz = matriz_confusao_top5(
        y_real=y_teste,
        y_pred=y_pred,
        nome_modelo=nome
    )

    matrizes_reduzidas[nome] = df_matriz

    print(df_matriz)

#===========================================
#   Organizar os modelos
#==========================================

modelos_finais = {
    "KNN": {
        "modelo": modelo_knn_final,
        "X": X_teste["B - Cor + Textura"]
    },
    "SVM": {
        "modelo": modelo_svm_final,
        "X": X_teste["D - Cor + Textura + Forma + HOG"]
    },
    "Random Forest": {
        "modelo": modelo_rf_final,
        "X": X_teste["C - Cor + Textura + Forma"]
    }
}

#===================================================
#   Identificação dos principais erros
#===================================================

def analisar_confusoes(modelo, X, y_real, nome_modelo):

    # Realizar previsões
    y_pred = modelo.predict(X)

    # Obter as classes
    classes = sorted(y_real.unique())

    # Calcular matriz de confusão
    matriz = confusion_matrix(
        y_real,
        y_pred,
        labels=classes
    )

    # Remover a diagonal principal (acertos)
    matriz_erros = matriz.copy()
    np.fill_diagonal(matriz_erros, 0)

    # Identificar os pares de classes confundidas
    pares = []

    for i in range(len(classes)):
        for j in range(len(classes)):

            if matriz_erros[i, j] > 0:
                pares.append({
                    "Modelo": nome_modelo,
                    "Classe real": classes[i],
                    "Classe predita": classes[j],
                    "Quantidade": matriz_erros[i, j]
                })

    # Organizar do maior para o menor número de erros
    df_pares = pd.DataFrame(pares)

    return df_pares.sort_values(
        "Quantidade",
        ascending=False
    )

#=============================================
#   Execução da comparação
#============================================

resultados_confusoes = []

for nome, dados in modelos_finais.items():

    df_confusoes = analisar_confusoes(
        modelo=dados["modelo"],
        X=dados["X"],
        y_real=y_teste,
        nome_modelo=nome
    )

    resultados_confusoes.append(df_confusoes)

    print(f"\nPrincipais confusões — {nome}")
    print(df_confusoes.head(10))
