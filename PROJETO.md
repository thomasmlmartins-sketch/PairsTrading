# Pairs Trading ML — Ibovespa
### Estratégia quantitativa automatizada com Machine Learning

---

## O que é este projeto?

Um sistema completo de **pairs trading** no mercado brasileiro (B3), com geração automática de sinais, envio de alertas por email e rastreamento de carteira fictícia.

O sistema:
1. Identifica pares de ações que "andam juntos" historicamente (cointegrados)
2. Detecta quando um par diverge além do normal
3. Usa Machine Learning (XGBoost) para confirmar se vale a pena operar
4. Gera sinais diários com ordens exatas (quantidade de ações, preços, risco)
5. Envia tudo por email às 18h30 automaticamente

---

## A lógica da estratégia

### O que é pairs trading?

Imagine dois ativos que historicamente se movem juntos — por exemplo, dois bancos grandes. Se um sobe muito mais que o outro sem motivo aparente, existe uma oportunidade: **vender o que subiu demais e comprar o que ficou para trás**, apostando que a relação vai se normalizar.

Essa é a essência do pairs trading: não é uma aposta direcional no mercado (não importa se vai subir ou cair), é uma aposta na **convergência de dois ativos**.

```
Spread = Preço_A − hedge_ratio × Preço_B

Situação normal:   spread ≈ 0  (os dois andam juntos)
Oportunidade:      spread >> 0  (A subiu demais) → vender A, comprar B
                   spread << 0  (B subiu demais) → comprar A, vender B
```

### O z-score

Para saber quando o spread está "longe demais" da média, usamos o **z-score**:

```
z = (spread − média_60_dias) / desvio_padrão_60_dias
```

- `|z| > 1.5` → spread divergiu o suficiente para considerar entrada
- `z = 0`     → spread voltou à média (sair da posição)
- `|z| > 2.5` → spread fugiu ainda mais (stop-loss, sair com prejuízo)

### O papel do XGBoost

O z-score por si só gera muitos falsos positivos — o spread pode continuar divergindo ao invés de reverter. Por isso usamos um modelo de Machine Learning para **filtrar as entradas**.

O XGBoost é treinado para responder: *"dado o estado atual do spread, qual a probabilidade de ele reverter para a média nos próximos 3 dias?"*

- Se `probabilidade > 0.45` **e** `|z| > 1.5` → abre posição
- Se só o z-score disparar mas o modelo discordar → aguarda

---

## Os 4 pares selecionados

Os pares foram escolhidos por dois critérios: cointegração estatística (teste ADF) e bom desempenho histórico no período de teste (2023–2026).

| Par | Relação econômica | Hedge Ratio |
|-----|------------------|-------------|
| **VALE3 / GGBR4** | Mineração e siderurgia: ambas expostas ao preço do minério de ferro | ~1.3 |
| **BBAS3 / TOTS3** | Banco + tecnologia financeira: TOTVS fornece software para bancos | ~0.8 |
| **BBDC4 / TOTS3** | Bradesco + TOTVS: mesma relação setorial | ~0.9 |
| **ITUB4 / EGIE3** | Itaú + Engie: diversificação entre financeiro e utilities | ~1.1 |

> O **hedge ratio** indica quantas ações de B comprar/vender para cada ação de A, de modo que o spread seja estacionário.

---

## Resultados do backtest (2023–2026)

> ⚠️ **Leia antes de se empolgar com os números**

Os resultados abaixo são do backtest sobre dados históricos. Eles **superestimam** o desempenho real por algumas razões:

1. **Os dados têm ~100 observações/ano** (não 252 como assumido no cálculo de CAGR). Isso infla o CAGR em ~3–4x. Um CAGR de 458% no backtest equivale a ~70–80% real.
2. **Custo de aluguel das ações** (short selling) não foi modelado — reduz ~1–4% ao ano por par.
3. **Seleção enviesada**: escolhemos os pares que funcionaram no período testado.
4. **Amostra pequena**: 10–22 trades por par é pouco para estatísticas robustas.

| Par | CAGR (backtest) | CAGR (estimado real) | Sharpe | Max Drawdown | Trades | Win Rate |
|-----|----------------|---------------------|--------|-------------|--------|----------|
| VALE3/GGBR4 | 458% | ~70–80% | 3.16 | -11.4% | 22 | 95.5% |
| BBAS3/TOTS3 | 437% | ~65–75% | 2.70 | -34.6% | 10 | 100% |
| BBDC4/TOTS3 | 349% | ~55–65% | 2.22 | -31.5% | 11 | 90.9% |
| ITUB4/EGIE3 | 184% | ~30–45% | 2.19 | -19.3% | 14 | 85.7% |

**Expectativa realista para o primeiro ano de operação real:**
- Pessimista: -20% a 0% (pares perdem cointegração)
- Base: +15% a +35% (estratégia funciona com atrito real)
- Otimista: +40% a +60% (próximo do backtest corrigido)

---

## Arquitetura do sistema

```
pairs_trading_ml/
│
├── data/
│   └── prices.xlsx              # Dados históricos da Economatica (B3)
│
├── results/
│   ├── xgb_model.json           # Modelo XGBoost treinado (salvo)
│   ├── cointegrated_pairs.csv   # Os 4 pares com hedge ratios
│   ├── backtest_summary.csv     # Métricas do backtest por par
│   ├── positions.json           # Posições reais abertas (rastreamento manual)
│   ├── paper_portfolio.json     # Carteira fictícia (atualizada todo dia)
│   ├── pending_orders.json      # Ordens a executar no dia seguinte (MT5)
│   ├── email_config.json        # Credenciais Gmail (App Password)
│   └── logs/
│       ├── signal_YYYY-MM-DD.txt   # Log diário dos sinais
│       └── execute_YYYY-MM-DD.txt  # Log diário das execuções
│
└── src/
    ├── data_loader.py      # Lê e limpa dados da Economatica
    ├── clustering.py       # K-Means para agrupar ações similares
    ├── cointegration.py    # Testa cointegração entre pares (ADF + Engle-Granger)
    ├── features.py         # Calcula z-score, RSI, momentum, half-life
    ├── model.py            # Treina e salva o XGBoost
    ├── backtest.py         # Simula operações com custo de transação
    ├── pipeline.py         # Orquestra todo o processo de treinamento
    ├── yfinance_loader.py  # Baixa preços atuais da B3 via internet
    ├── signal_generator.py # Gera sinais diários + position sizing + email
    ├── paper_trader.py     # Rastreia carteira fictícia
    ├── executor.py         # Executa ordens via MetaTrader 5 (Clear)
    ├── execute_orders.py   # Script matinal de execução automática
    ├── notifier.py         # Envia emails via Gmail
    └── stress_test.py      # Testa grade de parâmetros para otimização
```

---

## Como o pipeline foi construído (passo a passo)

### Etapa 1 — Dados
- Fonte: Economatica (fechamento ajustado por proventos, ~100 observações/ano)
- Período: 2016–2026
- Tickers: todos os principais do Ibovespa (~50 ações)
- Limpeza: forward fill (máx. 5 dias), remoção de tickers com menos de 252 observações

### Etapa 2 — Clustering K-Means
- Agrupa ações com comportamentos similares (k=8 clusters)
- Usa PCA para reduzir dimensionalidade antes do clustering
- Gera ~200 pares candidatos (apenas pares do mesmo cluster são testados)
- Objetivo: reduzir o espaço de busca e evitar pares sem relação econômica

### Etapa 3 — Filtro de Cointegração
- Para cada par candidato, testa se o spread é estacionário (ADF + Engle-Granger)
- Janela de formação: 500 observações (~5 anos de dados)
- Threshold: p-value < 0.10 no ADF (mais relaxado que o padrão 0.05)
- Resultado: 4 pares passaram no filtro

### Etapa 4 — Features para o XGBoost
Calculadas apenas com dados passados (sem data leakage):

| Feature | Descrição |
|---------|-----------|
| `zscore` | Distância atual da média (janela 60 dias) |
| `zscore_abs` | Valor absoluto do z-score |
| `zscore_lag1` | Z-score do dia anterior |
| `zscore_lag5` | Z-score de 5 dias atrás |
| `spread_vol` | Volatilidade rolling do spread (20 dias) |
| `spread_rsi` | RSI do spread (14 dias) |
| `spread_momentum_5` | Variação do spread em 5 dias |
| `spread_momentum_10` | Variação do spread em 10 dias |

**Label (o que o modelo aprende a prever):**
> "O z-score vai cruzar ±0.3 nos próximos 3 dias?"
- 1 = sim (spread vai reverter) → abrir posição
- 0 = não → aguardar

### Etapa 5 — Treinamento do XGBoost
- Split temporal: treino ≤ 2021, validação 2022, teste 2023+
- Filtro de treinamento: só treina em amostras onde `|z| > 1.0` (spread já divergiu)
- Resultado: AUC = 0.57 na validação (acima do acaso, mas modesto)
- O modelo **filtra** mais do que **prevê** — o z-score faz o trabalho principal

### Etapa 6 — Backtest
- Regras de entrada: `|z| > 1.5` **E** `prob > 0.45`
- Regra de saída: `z * side ≥ 0` (spread reverteu) **OU** `|z| > 2.5` (stop-loss)
- Custo de transação: 0.15% por operação (ida + volta = 0.30%)
- Position sizing: igual para todos os pares

---

## Parâmetros de operação

Todos alinhados entre o backtest e o sistema de sinais ao vivo:

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `entry_zscore` | ±1.5 | Mínimo para considerar entrada |
| `exit_zscore` | 0.0 | Sair quando spread cruzar a média |
| `stop_zscore` | ±2.5 | Stop-loss se spread piorar |
| `xgb_threshold` | 0.45 | Prob. mínima do XGBoost para confirmar |
| `zscore_window` | 60 obs | Janela para calcular z-score ao vivo |
| `transaction_cost` | 0.15% | Custo por operação |

---

## Automação diária

O sistema roda automaticamente sem nenhuma intervenção:

```
18:30 (seg a sex)   — Windows Task Scheduler dispara run_signals.bat
│
├─ Baixa preços frescos da B3 via yfinance
├─ Calcula z-score e probabilidade XGBoost para cada par
├─ Determina ação: AGUARDAR / ABRIR / MANTER / FECHAR
├─ Calcula position sizing (nº de ações, capital, risco no stop)
├─ Atualiza carteira fictícia (paper portfolio)
├─ Salva ordens pendentes em pending_orders.json
└─ Envia email para thomasmlmartins@gmail.com

10:05 (dia seguinte) — Windows Task Scheduler dispara run_execute.bat
│
├─ Lê pending_orders.json
├─ Conecta ao MetaTrader 5 (Clear Corretora, precisa estar aberto)
├─ Executa ordens a mercado para ambos os legs
├─ Atualiza positions.json com preços reais de execução
└─ Envia email de confirmação: "Execucao OK" ou "ERRO"
```

---

## O email diário

Todo dia útil às 18h30 você recebe um email com:

**1. Tabela de sinais**
```
Par              Z-score   Prob    Posição       Ação
VALE3/GGBR4      -1.549   0.248      FLAT     AGUARDAR
BBDC4/TOTS3      -1.666   0.542      FLAT     ABRIR LONG SPREAD
```

**2. Ordens detalhadas (quando há sinal)**
```
[BBDC4/TOTS3]  ABRIR LONG SPREAD
  COMPRAR   BBDC4:  1.168 ações @ R$ 17,65  =  R$ 20.615
  VENDER    TOTS3:    138 ações @ R$ 31,67  =  R$  4.370
  Capital alocado : R$ 24.986
  Risco no stop   : R$  1.062  (4.2% do capital do par)
```

**3. Carteira fictícia (paper trading)**
```
CARTEIRA FICTICIA (Paper Trading)
  Capital inicial : R$ 100.000,00
  Valor atual     : R$ 107.340,00
  Retorno total   : +7.34%
  Trades fechados : 3

  Posicoes abertas:
    BBDC4/TOTS3  desde 2026-05-13  P&L: +R$ 840 (+3.4%)

  Ultimos trades fechados:
    OK VALE3/GGBR4  2026-05-02 → 2026-05-08  +R$ 2.340 (+9.3%)
    OK BBAS3/TOTS3  2026-04-15 → 2026-04-22  +R$ 1.100 (+4.4%)
    XX ITUB4/EGIE3  2026-04-01 → 2026-04-03  -R$  560 (-2.2%)
```

---

## Como rodar manualmente

```bash
# Ver sinais do dia (sem enviar email)
python src/signal_generator.py --capital 100000

# Ver sinais e enviar email
python src/signal_generator.py --capital 100000 --email

# Ver sinais, enviar email e registrar posições abertas
python src/signal_generator.py --capital 100000 --email --update-positions --save-orders

# Executar ordens pendentes via MT5 (simulado)
python src/execute_orders.py --dry-run

# Executar ordens reais via MT5
python src/execute_orders.py

# Configurar email (primeira vez)
python src/notifier.py --setup

# Rodar o pipeline completo de treinamento
python src/pipeline.py --prices data/prices.xlsx --formation-window 500 --pvalue-threshold 0.10

# Stress test de parâmetros
python src/stress_test.py
```

---

## Tarefas agendadas (Windows Task Scheduler)

| Tarefa | Horário | Script |
|--------|---------|--------|
| `PairsTrading_Sinais` | 18:30 seg–sex | `run_signals.bat` |
| `PairsTrading_Execucao` | 10:05 seg–sex | `run_execute.bat` |

Para verificar:
```powershell
Get-ScheduledTask -TaskName "PairsTrading*"
```

---

## Requisitos para execução real

- [ ] Conta na **Clear Corretora** (aberta e aprovada)
- [ ] **MetaTrader 5** instalado com conta Clear
- [ ] Trading automático ativado no MT5 (`Ferramentas → Opções → Expert Advisors`)
- [ ] Capital depositado na Clear
- [ ] Computador **ligado e com MT5 aberto** às 10:05 (ou VPS Windows)
- [ ] `pip install MetaTrader5` executado

---

## Bugs corrigidos durante o desenvolvimento

Quatro bugs críticos foram identificados e corrigidos — todos os resultados produzidos antes das correções eram inválidos:

| Bug | Local | Problema | Correção |
|-----|-------|----------|----------|
| Exit impossível | `backtest.py` | `abs(z) < 0.0` é sempre False — posições nunca saíam | Corrigido para `z * side ≥ -exit_z` |
| Label trivial | `pipeline.py` | 93–100% das labels eram 1, modelo inútil | Filtro `\|z\| > 1.0` + horizon=3 + threshold=0.3 |
| n_trades errado | `backtest.py` | Contava dias, não trades | Usa transições 0→não-zero da série de posição |
| argparse override | `pipeline.py` | Defaults do argparse sobrescreviam Config | Alinhamento de todos os defaults |

---

## Próximos passos sugeridos

### Curto prazo (dentro deste projeto)
- [ ] **Regime detection**: detectar mercado em tendência e pausar operações
- [ ] **Correção do CAGR**: ajustar `compute_metrics` para frequência real dos dados (~100/ano)
- [ ] **VPS Windows**: manter o MT5 rodando 24/7 sem depender do computador pessoal

### Médio prazo (novos projetos)
- [ ] **Factor Investing**: estratégia separada usando dados fundamentalistas (P/L, ROE, momentum)
- [ ] **ADR Arbitrage**: explorar divergências entre VALE3 (B3) e VALE (NYSE)
- [ ] **Basket trading**: cointegração entre grupos de 3–4 ações (mais robusto que pares)

---

## Stack tecnológica

| Componente | Tecnologia |
|------------|-----------|
| Linguagem | Python 3.12 |
| Dados históricos | Economatica (Excel) |
| Dados ao vivo | yfinance |
| Clustering | scikit-learn (K-Means + PCA) |
| Cointegração | statsmodels (ADF, OLS) |
| Machine Learning | XGBoost |
| Execução | MetaTrader 5 (Python API) |
| Notificação | Gmail SMTP |
| Agendamento | Windows Task Scheduler |

---

*Última atualização: maio de 2026*
