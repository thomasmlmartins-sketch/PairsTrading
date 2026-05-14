# Pairs Trading ML — Ibovespa

Estratégia de pairs trading no Ibovespa usando clustering K-Means para seleção de pares e XGBoost para classificação de reversão do spread, com backtest realista incluindo custo de transação.

## Estrutura de pastas

```
pairs_trading_ml/
├── data/           # CSVs exportados da Economatica (preços de fechamento ajustado)
├── notebooks/      # Exploração, análise e visualizações
├── results/        # Outputs: métricas de backtest, gráficos, pares selecionados
└── src/            # Código-fonte modularizado
```

## Stack

- **Python 3.10+**
- **Dados**: pandas, numpy
- **ML**: scikit-learn (K-Means, PCA, StandardScaler), xgboost
- **Cointegração / spread**: statsmodels (ADF, Johansen, OLS)
- **Backtest**: lógica própria em `src/backtest.py`
- **Visualização**: matplotlib, seaborn

## Fonte de dados

Arquivos CSV exportados da **Economatica** com preços de fechamento ajustado por proventos.

- Formato esperado: colunas = tickers, linhas = datas (índice `datetime`)
- Caminho padrão: `data/prices.csv`
- Os arquivos de dados **não são versionados** (adicionar ao `.gitignore`)

## Pipeline principal

### 1. Pré-processamento (`src/data_loader.py`)
- Leitura e limpeza dos CSVs da Economatica
- Tratamento de missing values (forward fill com limite de 5 dias)
- Cálculo de log-retornos e preços normalizados

### 2. Clustering K-Means (`src/clustering.py`)
- Features: correlação de retornos, setor GICS, beta de mercado
- PCA para redução de dimensionalidade antes do clustering
- Seleção do `k` ótimo via silhouette score e elbow method
- Output: pares candidatos dentro do mesmo cluster

### 3. Filtragem por cointegração (`src/cointegration.py`)
- Teste ADF no spread de cada par candidato
- Teste de Johansen como validação adicional
- Threshold: p-value < 0.05
- Período de formação: janela deslizante (padrão: 252 dias úteis)

### 4. Features e label para XGBoost (`src/features.py`)
- **Features do spread**: z-score, half-life (Ornstein-Uhlenbeck), volatilidade rolling, RSI do spread, distância do z-score da média
- **Label**: reversão do spread nos próximos `N` dias (binário: 1 = reverteu para dentro de ±0.5σ)
- Split temporal: treino até 2021, validação 2022, teste 2023+

### 5. Modelo XGBoost (`src/model.py`)
- Classificação binária: probabilidade de reversão do spread
- Threshold de entrada configurável (padrão: `prob > 0.6`)
- Sem data leakage: features calculadas apenas com dados passados
- Métricas: AUC-ROC, precision, recall, F1

### 6. Backtest (`src/backtest.py`)
- Sinal de entrada: z-score > 2 (ou < -2) **e** XGBoost prob > threshold
- Sinal de saída: z-score cruza zero ou stop-loss em ±3σ
- **Custo de transação**: 0.15% por operação (ida + volta = 0.30%), configurável
- Position sizing: igual para os dois legs do par
- Métricas: Sharpe ratio, max drawdown, CAGR, % operações lucrativas, turnover

## Convenções de código

- Funções puras sempre que possível — sem side effects implícitos
- Parâmetros configuráveis via dicionário `config` ou dataclass, nunca hardcoded no meio das funções
- Logs com `logging` (não `print`) em código de produção; `print` aceitável em notebooks
- Tipos anotados nas assinaturas públicas
- Nenhum notebook importa de outro notebook — toda lógica reutilizável vai para `src/`

## Parâmetros padrão

| Parâmetro | Valor padrão | Descrição |
|-----------|-------------|-----------|
| `formation_window` | 252 | Dias úteis para calibrar cointegração e modelo |
| `trading_window` | 63 | Dias úteis para operar cada par |
| `entry_zscore` | 2.0 | Z-score mínimo para abrir posição |
| `exit_zscore` | 0.0 | Z-score de saída (cruzamento da média) |
| `stop_zscore` | 3.0 | Stop-loss em z-score |
| `xgb_threshold` | 0.60 | Prob. mínima do XGBoost para confirmar entrada |
| `transaction_cost` | 0.0015 | Custo por operação (0.15%) |
| `k_clusters` | 8 | Número de clusters K-Means (ajustar via silhouette) |

## Reprodutibilidade

- Fixar `random_state=42` em K-Means e XGBoost
- Fixar `numpy.random.seed(42)` no início dos scripts de treinamento
- Salvar modelo treinado em `results/xgb_model.json` com `model.save_model()`

## O que não fazer

- Não usar dados futuros para calcular features (data leakage)
- Não otimizar hiperparâmetros no período de teste
- Não ignorar custos de transação no backtest — distorce completamente o resultado
- Não misturar períodos de formação e trading na mesma janela
