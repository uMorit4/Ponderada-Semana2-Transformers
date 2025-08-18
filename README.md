# Tradução Automatizada com Transformers

O repositório contém uma implementação de um modelo com a técnica de Transformers para traduzir frases do português para o ingles, usando de base o dataset do TED

Para acessar o código, acesse: transformer.ipynb

## Treinamento do modelo e comparativo de desempenho

O modelo utilizado foi o seq2seq do tipo Transformer encoder–decoder, com encoder de pilha de blocos com multi-head self-attention + MLP,residual + LayerNorm e decoder com masked self-attention (look-ahead) + cross-attention ao encoder, MLP, residual + LayerNorm

(OBS: Não foi possível executar o tutorial até o final, visto que o google colab teve limitação de tempo para uso da T4 e crashou depois de um tempo em CPU)

## TABELA COMPARATIVA

| Aspecto                   | CPU                                                                                                                                              | GPU                                                                                                      |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Eficiência computacional  | Bem mais lenta; processamento mais sequencial; operações com matrizes muito mais custosas; **tempo por step maior**.                             | Muito mais **eficaz**, usando **paralelismo** para acelerar os cálculos por época.                        |
| Uso de memória            | **Maior**; forte dependência da **RAM** do Colab para conseguir realizar os cálculos.                                                            | Usa a **VRAM** da placa, ajudando a “aliviar” o uso de RAM do sistema.                                   |
| Necessidade de hardware   | Acesso amplamente disponível e **gratuita** no Google Colab.                                                                                     | **Limitada** no free-tier; disponibilidade de **T4** por tempo restrito.                                 |


# Percepções pessoais:

**Pontos positivos**:

O notebook é bem organizado em etapas (dados → modelo → loss mascarada → treino → inferência), o que facilita entender e adaptar.

As explicações visuais (positional encoding, atenção) ajudam a ligar teoria e prática.

O pipeline com tf.data e prefetch(AUTOTUNE) já dá um empurrão de desempenho.

O uso de loss mascarada (ignora padding) é correto para seq2seq e evita enviesar as métricas.

**Pontos negativos**:

Dependências sensíveis a versão (ex.: tensorflow-text) e confusão com o path do SavedModel do tokenizador; é preciso apontar para a pasta do modelo, não para o arquivo .pb.

No Colab sem GPU, o treino fica muito lento; mesmo com T4 ainda é pesado se batch_size/max_length estiverem altos.

Falta de callbacks prontos (ex.: EarlyStopping, ModelCheckpoint) faz o treino durar mais do que o necessário.

Facilmente dá erro de memória se os steps e os comprimentos de sequência forem grandes.

**Outras observações**:

O treinamento usa teacher forcing; é importante sempre checar a inferência autoregressiva (greedy/beam) para ver qualidade real das traduções.

Ativar mixed precision na GPU T4 costuma dar ganho de tempo sem piorar a qualidade.

Manter apenas uma pasta do conversor/tokenizer (evitar pastas duplicadas) evita erros de carregamento.

# Próximos passos

**Ambiente**:

- Fixar versões em requirements.txt (TF, TFDS, TF-Text compatíveis).

- Confirmar GPU ativa (tf.config.list_physical_devices('GPU')) e ligar mixed precision.

**Treino**:

- Começar com epochs e max_length menores para validar o fluxo; depois escalar.

- Adicionar EarlyStopping(patience=2) e ModelCheckpoint para não desperdiçar tempo.

- Usar scheduler de learning rate com warmup.

**Avaliação**:

- Separar validação fixa e medir qualidade com SacreBLEU.

- Salvar exemplos PT→EN antes/depois para comparação qualitativa.

**Modelo**:

- Criar um baseline “pequeno” (menos camadas/heads) para iteração rápida.

- Testar label_smoothing e dropout moderado para generalização.

- Reprodutibilidade e comparação CPU vs GPU

- Fixar seeds, registrar tempo por época (callback) e total.


