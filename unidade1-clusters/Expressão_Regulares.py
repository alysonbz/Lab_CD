import re
from datetime import datetime
import pandas as pd

# ============================================================
# 1. PADRÕES REGEX COMPILADOS (TODAS AS COLUNAS)
# ============================================================
RX_ESPACES = re.compile(r"\s+")
RX_CPF = re.compile(r"(?<!\d)(?:\d{3}\.?\d{3}\.?\d{3}-?\d{2})(?!\d)")
RX_CNPJ = re.compile(r"(?<!\d)(?:\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2})(?!\d)")
RX_EMAIL_VALIDO = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
RX_DATA_EXTRAIR = re.compile(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})")
RX_CEP = re.compile(r"(?<!\d)\d{5}-?\d{3}(?!\\d)")
RX_DIGITOS_FONE = re.compile(r"\d+")

# ============================================================
# 2. FUNÇÕES DE VALIDAÇÃO MATEMÁTICA (MÓDULO 11 & CALENDÁRIO)
# ============================================================
def cpf_valido(cpf: str) -> bool:
    if pd.isna(cpf): return False
    nums = re.sub(r"\D", "", str(cpf))
    if len(nums) != 11 or len(set(nums)) == 1:
        return False

    def digito(parcial):
        soma = sum(int(d) * w for d, w in zip(parcial, range(len(parcial) + 1, 1, -1)))
        r = (soma * 10) % 11
        return '0' if r == 10 else str(r)

    d1 = digito(nums[:9])
    d2 = digito(nums[:9] + d1)
    return nums[-2:] == d1 + d2


def cnpj_valido(cnpj: str) -> bool:
    if pd.isna(cnpj): return False
    nums = re.sub(r"\D", "", str(cnpj))
    if len(nums) != 14 or len(set(nums)) == 1:
        return False
    pesos1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    pesos2 = [6] + pesos1

    def calc(parcial, pesos):
        s = sum(int(d) * p for d, p in zip(parcial, pesos))
        r = s % 11
        return '0' if r < 2 else str(11 - r)

    d1 = calc(nums[:12], pesos1)
    d2 = calc(nums[:12] + d1, pesos2)
    return nums[-2:] == d1 + d2

# ============================================================
# 3. FUNÇÕES DE NORMALIZAÇÃO E HIGIENIZAÇÃO COM REGEX
# ============================================================
def limpa_nome(texto: str) -> str:
    if pd.isna(texto): return None
    texto_limpo = RX_ESPACES.sub(" ", str(texto)).strip()
    return texto_limpo.title()


def normaliza_cpf(texto: str) -> str:
    if pd.isna(texto): return None
    nums = re.sub(r"\D", "", str(texto))
    if len(nums) == 11:
        return f"{nums[:3]}.{nums[3:6]}.{nums[6:9]}-{nums[9:]}"
    return str(texto)


def normaliza_cnpj(texto: str) -> str:
    if pd.isna(texto): return None
    nums = re.sub(r"\D", "", str(texto))
    if len(nums) == 14:
        return f"{nums[:2]}.{nums[2:5]}.{nums[5:8]}/{nums[8:12]}-{nums[12:]}"
    return str(texto)


def normaliza_fone(texto: str) -> str:
    if pd.isna(texto): return None
    nums = re.sub(r"\D", "", str(texto))
    if nums.startswith("55") and len(nums) > 10:
        nums = nums[2:]
    if len(nums) == 11:
        return f"({nums[:2]}) {nums[2:7]}-{nums[7:]}"
    elif len(nums) == 10:
        return f"({nums[:2]}) 9{nums[2:6]}-{nums[6:]}"
    return str(texto)


def limpa_e_valida_email(texto: str) -> tuple[str, bool]:
    if pd.isna(texto): return None, False
    email_limpo = str(texto).strip().lower().replace(",", ".")
    valido = bool(RX_EMAIL_VALIDO.match(email_limpo))
    return email_limpo, valido


def normaliza_e_valida_cep(texto: str) -> tuple[str, bool]:
    if pd.isna(texto): return None, False
    nums = re.sub(r"\D", "", str(texto))
    if len(nums) == 8:
        return f"{nums[:5]}-{nums[5:]}", True
    return str(texto), False


def normaliza_e_valida_data(texto: str) -> tuple[str, bool]:
    if pd.isna(texto): return None, False
    texto_str = str(texto).strip()
    match = RX_DATA_EXTRAIR.search(texto_str)
    if match:
        d, m, a = match.groups()
        data_norm = f"{int(d):02d}/{int(m):02d}/{a}"
        try:
            datetime.strptime(data_norm, "%d/%m/%Y")
            return data_norm, True
        except ValueError:
            return data_norm, False
    return texto_str, False


def normaliza_valor(texto: str) -> str:
    if pd.isna(texto): return None
    apenas_num = "".join(re.findall(r"[\d,.]", str(texto)))
    if apenas_num:
        return f"R$ {apenas_num.strip()}"
    return str(texto)


# ============================================================
# 4. PIPELINE CENTRAL DE PROCESSAMENTO E AUDITORIA
# ============================================================
def processar_dataset_completo(caminho_arquivo: str):
    print(f"Lendo o arquivo local: '{caminho_arquivo}'...\n")
    df = pd.read_csv(caminho_arquivo)
    df_original = df.copy()  # Mantido para fins de comparação e auditoria
    total = len(df)

    # Execução sequencial de Higienização e Auditoria
    df['nome_limpo'] = df['nome'].apply(limpa_nome)

    df['cpf_normalizado'] = df['cpf'].apply(normaliza_cpf)
    df['cpf_valido'] = df['cpf_normalizado'].apply(cpf_valido)

    df['cnpj_normalizado'] = df['cnpj'].apply(normaliza_cnpj)
    df['cnpj_valido'] = df['cnpj_normalizado'].apply(cnpj_valido)

    df['telefone_normalizado'] = df['telefone'].apply(normaliza_fone)

    emails_processados = df['email'].apply(limpa_e_valida_email)
    df['email_normalizado'] = [e[0] for e in emails_processados]
    df['email_valido'] = [e[1] for e in emails_processados]

    ceps_processados = df['cep'].apply(normaliza_e_valida_cep)
    df['cep_normalizado'] = [c[0] for c in ceps_processados]
    df['cep_valido'] = [c[1] for c in ceps_processados]

    datas_processadas = df['data'].apply(normaliza_e_valida_data)
    df['data_normalizada'] = [d[0] for d in datas_processadas]
    df['data_valida'] = [d[1] for d in datas_processadas]

    df['valor_normalizado'] = df['valor'].apply(normaliza_valor)
    df['valor_valido'] = df['valor_normalizado'].apply(lambda x:
                    bool(x and re.search(r'\d', str(x))))

    # Consolidação das Métricas Finais
    cpfs_ok = df['cpf_valido'].sum()
    cnpjs_ok = df['cnpj_valido'].sum()
    emails_ok = df['email_valido'].sum()
    ceps_ok = df['cep_valido'].sum()
    datas_ok = df['data_valida'].sum()
    valor_ok = df['valor_valido'].sum()

    nomes_alterados = (df['nome'] != df['nome_limpo']).sum()
    cpfs_alterados = (df['cpf'] != df['cpf_normalizado']).sum()
    cnpjs_alterados = (df['cnpj'] != df['cnpj_normalizado']).sum()
    fones_alterados = (df['telefone'] != df['telefone_normalizado']).sum()
    valores_alterados = (df['valor'] != df['valor_normalizado']).sum()

    # Impressão Estruturada do Relatório Unificado
    print("=" * 60)
    print("        RELATÓRIO UNIFICADO DE QUALIDADE E LIMPEZA")
    print("=" * 60)
    print(f"Número total de registros analisados: {total}")
    print("-" * 60)
    print(f"Percentual de CPFs válidos (Módulo 11):   {(cpfs_ok / total) * 100:.2f}%")
    print(f"Percentual de CNPJs válidos (Módulo 11):  {(cnpjs_ok / total) * 100:.2f}%")
    print(f"Percentual de E-mails válidos estrutural: {(emails_ok / total) * 100:.2f}%")
    print(f"Percentual de CEPs válidos (8 dígitos):   {(ceps_ok / total) * 100:.2f}%")
    print(f"Percentual de Datas válidas (Calendário): {(datas_ok / total) * 100:.2f}%")
    print(f"Percentual de Valor:                      {(valor_ok / total) * 100:.2f}%")
    print("-" * 60)
    print("Volume de strings modificadas/corrigidas por campo:")
    print(f"  • NOME:     {nomes_alterados} registros formatados")
    print(f"  • CPF:      {cpfs_alterados} registros formatados")
    print(f"  • CNPJ:     {cnpjs_alterados} registros formatados")
    print(f"  • TELEFONE: {fones_alterados} registros formatados")
    print(f"  • VALOR:    {valores_alterados} registros formatados")
    print("=" * 60)

    print("\nExemplos práticos de transformações aplicadas (Antes -> Depois):")
    amostra_idx = df[df['cpf'] != df['cpf_normalizado']].index[:2]
    for idx in amostra_idx:
        print(f"  [CPF]  {df_original.loc[idx, 'cpf']}  -->  {df.loc[idx, 'cpf_normalizado']}")

    amostra_fone_idx = df[df['telefone'] != df['telefone_normalizado']].index[:2]
    for idx in amostra_fone_idx:
        print(f"  [TELEFONE] {df_original.loc[idx, 'telefone']}  -->  {df.loc[idx, 
        'telefone_normalizado']}")

    print("=" * 60)

    return df


# ============================================================
# EXECUÇÃO DO CONJUNTO
# ============================================================

df_final = processar_dataset_completo("dataset_sintetico_regex_ruidos.csv")
