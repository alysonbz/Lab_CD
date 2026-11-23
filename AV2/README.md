# Análise do dataset TrashNet

## Instalação
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Execução
Na raiz do projeto:
```bash
python main.py
```

O dataset é baixado automaticamente na primeira execução. Os dados e gráficos são salvos em `data/` e `outputs/`.
Para usar um dataset já baixado, coloque a pasta `dataset-resized` dentro de `data/`.
