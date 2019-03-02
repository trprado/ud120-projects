# Projeto Final Fundamentos de Data Science II - Udacity

Os códigos python do projeto final e mini projetos foram todos atualizados para python 3, incluindo os códigos dos arquivos `poi_id.py` e `tester.py`

- O arquivo principal das análises é o `poi_id.py`, algumas linhas de código se encontram comentadas, caso sejam executadas pode haver uma grande carga de processamento e tempo necessário.
- O arquivo `tools_fp.py` contém algumas funções de ajuda para a execução do arquivo principal. Nele existe três ferramentas que são para seleção das *features*, remoção de *outliers* das *features* e geração de novas *features*.
- O arquivo `Report.md` contém o texto da resposta as pergunta exigidas junto ao projeto final.
- O arquivo `referencias.md` contém link das referências usadas para desenvolver o projeto final.
- Caso as funções de `tools_fp.py` sejam executadas com a opção de plotar os gráficos, esses serão gravados em uma pasta `fig` no formato `.png`.
- O arquivo `poi_id.py` gera alguns arquivos com resultados das pontuações da validação cruzada, esses arquivos são gravados na pasta `results` com o formato `.html`.

## Requisitos

Esse projeto foi criado usando [Python 3](www.python.org) em sua versão 3.6, junto também foi utilizado o gerenciador de pacotes e dependências [Poetry](https://poetry.eustace.io). Porém um arquivo `requeriments.txt` foi gerado com todos os módulos utilizados para ser utilizado com o PIP.

Para instalar as dependências com Poetry:

```bash
$ poetry install
```

Para instalar as dependências com PIP:
```bash
$ pip install -r requirements.txt
```

## Execução

Para executar o `poi_id.py`

```bash
$ python poi_id.py
```

> **Obs.:** Esse processo pode ser demorado.

Para executar o `tester.py`

```bash
$ python tester.py
```

> **Obs.:** Esse processo pode ser demorado.