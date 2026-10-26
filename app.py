import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Radar de IA Avançado", layout="wide")

# =========================================================================
# 🛡️ CONFIGURAÇÕES DE SEGURANÇA E NOTIFICAÇÕES
# Substitua a palavra 'minhasenha123' pela sua senha definitiva se quiser
SENHA_CORRETA = "minhasenha123"
# =========================================================================

def enviar_notificacao_telegram(token, chat_id, mensagem):
    if token and chat_id:
        url = f"https://telegram.org{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=5)
        except:
            pass

# Interface de login visual na barra lateral
st.sidebar.title("🔐 Área Restrita")
senha_digitada = st.sidebar.text_input("Digite a senha para acessar a IA:", type="password")

if senha_digitada == SENHA_CORRETA:
    st.title("🤖 IA Quantitativa & Gestão de Risco v3.0")
    st.subheader("Análise Multi-Timeframe, Position Sizing Dinâmico e Alertas Mobile")

    # NOVOS CAMPOS PARA NOTIFICAÇÃO DO TELEGRAM (Adicionados aqui na lateral)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📢 Configurar Alertas no Celular")
    telegram_token = st.sidebar.text_input("Token do Bot do Telegram (do BotFather):", type="password", value="")
    telegram_chat_id = st.sidebar.text_input("Chat ID do Telegram (do UserInfoBot):", type="password", value="")

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Gerenciamento de Capital")
    capital_total = st.sidebar.number_input("Seu Capital Total Disponível (R$):", value=1000.0, step=100.0)
    risco_max_operacao = st.sidebar.number_input("Risco Máximo Aceito por Operação (R$):", value=100.0, step=10.0)

    st.write("### 🔍 Consultar uma Ação Específica")
    ticker_pesquisado = st.text_input("Digite o código da ação (ex: PETR4.SA, VALE3.SA, AAPL):", value="").strip().upper()

    LISTA_ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BBDC4.SA", "MGLU3.SA", "ABEV3.SA", "WEGE3.SA"]
    if ticker_pesquisado and ticker_pesquisado not in LISTA_ACOES:
        LISTA_ACOES.insert(0, ticker_pesquisado)

    @st.cache_data(ttl=600)
    def processar_ia_avancada(lista_ativos, cap, risco_max, tok, cid):
        resultados = []
        for ticker in lista_ativos:
            try:
                dados = yf.download(ticker, start="2023-01-01", progress=False)
                if dados is None or dados.empty or len(dados) < 200: 
                    continue
                
                if isinstance(dados.columns, pd.MultiIndex):
                    dados.columns = dados.columns.get_level_values(0)

                # INDICADORES TEMPORAIS
                dados['Retorno'] = dados['Close'].pct_change()
                dados['Media_Fast'] = dados['Close'].rolling(window=5).mean() / dados['Close']
                dados['Media_Trend'] = dados['Close'].rolling(window=20).mean() / dados['Close']
                dados['Media_Macro'] = dados['Close'].rolling(window=200).mean() / dados['Close']
                dados['Volatilidade'] = dados['Retorno'].rolling(window=10).std()
                dados['Volume_Var'] = dados['Volume'].pct_change()

                dados['Alvo'] = np.where(dados['Close'].shift(-1) > dados['Close'], 1, 0)
                dados = dados.dropna()

                if len(dados) < 10:
                    continue

                recursos = ['Media_Fast', 'Media_Trend', 'Media_Macro', 'Volatilidade', 'Retorno', 'Volume_Var']
                X = dados[recursos]
                y = dados['Alvo']

                divisao = int(len(X) * 0.8)
                X_treino, X_teste = X.iloc[:divisao], X.iloc[divisao:]
                y_treino, y_teste = y.iloc[:divisao], y.iloc[divisao:]

                modelo = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
                modelo.fit(X_treino, y_treino)

                previsoes_teste = modelo.predict(X_teste)
                acuracia = accuracy_score(y_teste, previsoes_teste) * 100

                ultimos_dados = X.tail(1)
                preco_atual = float(dados['Close'].iloc[-1])
                probabilidades = modelo.predict_proba(ultimos_dados)
                probabilidade_alta = float(probabilidades * 100)
                
                stop_loss = preco_atual * 0.97
                preco_venda_alvo = preco_atual * 1.09

                perda_por_acao = preco_atual - stop_loss
                quantidade_lote = int(risco_max / perda_por_acao)
                
                if (quantidade_lote * preco_atual) > cap:
                    quantidade_lote = int(cap // preco_atual)

                macro_alta = float(dados['Close'].iloc[-1]) > float(dados['Close'].rolling(window=200).mean().iloc[-1])
                
                if probabilidade_alta >= 60.0 and acuracia > 50.0 and macro_alta:
                    veredicto = "🎯 COMPRAR"
                    estrategia = "Entrada Rápida Diária/Semanal (Tendência Macro Anual de Alta Confirmada)"
                    
                    # Disparar Alerta se configurado
                    msg = f"🚨 *SINAL DE COMPRA IA* 🚨\nAtivo: {ticker}\nPreço: R$ {preco_atual:.2f}\nQtd Recomendada: {quantidade_lote} ações\nAlvo de Venda: R$ {preco_venda_alvo:.2f}\nStop Loss: R$ {stop_loss:.2f}"
                    enviar_notificacao_telegram(tok, cid, msg)
                else:
                    veredicto = "⏳ FIQUE DE FORA"
                    quantidade_lote = 0
                    stop_loss = 0.0
                    preco_venda_alvo = 0.0
                    estrategia = "Aguardar melhor momento técnico nos tempos gráficos de controle"

                resultados.append({
                    "Ação": ticker,
                    "Preço Atual": f"R$ {preco_atual:.2f}",
                    "Taxa de Acerto (IA)": f"{acuracia:.1f}%",
                    "Certeza de Alta": f"{probabilidade_alta:.1f}%",
                    "Veredito": veredicto,
                    "Qtd p/ Comprar (Lote)": quantidade_lote,
                    "Preço de Venda (Alvo)": f"R$ {preco_venda_alvo:.2f}" if preco_venda_alvo > 0 else "N/A",
                    "Stop Loss (Proteção)": f"R$ {stop_loss:.2f}" if stop_loss > 0 else "N/A",
                    "Estratégia de Tempo": estrategia
                })
            except:
                continue
                
        return pd.DataFrame(resultados)

    df_painel = processar_ia_avancada(LISTA_ACOES, capital_total, risco_max_operacao, telegram_token, telegram_chat_id)

    if df_painel.empty:
        st.warning("⚠️ Aguardando resposta da API financeira. Atualize a página em alguns instantes.")
    else:
        col1, col2, col3 = st.columns(3)
        oportunidades_hoje = len(df_painel[df_painel["Veredito"] == "🎯 COMPRAR"]) if "Veredito" in df_painel.columns else 0
        col1.metric("Ações Ativas Online", len(df_painel))
        col2.metric("Oportunidades de Compra", oportunidades_hoje)
        col3.metric("Risco Máx por Ativo", f"R$ {risco_max_operacao:.2f}")

        st.write("### 📋 Painel de Execução Tática Geral")
        st.dataframe(df_painel, use_container_width=True)
        
        st.markdown("---")
        st.write("### 📅 Matriz de Estratégia de Tempos Gráficos")
        st.info("""
        • **Planejamento Diário (Gatilho):** Disparado via momentum de 5 dias e variação rápida de volume. Estipula o Stop Loss estrito em 3%.
        • **Planejamento Semanal (Filtro):** Alinhado com a Média Móvel de 20 períodos para garantir que a operação não vá contra a tendência da semana.
        • **Planejamento Anual (Proteção):** Filtrado de forma absoluta pela Média Macro de 200 períodos. O sistema bloqueia compras automaticamente se a ação estiver barata mas em colapso de longo prazo.
        """)

elif senha_digitada != "":
    st.sidebar.error("❌ Senha Incorreta!")
else:
    st.info("🔒 Sistema Bloqueado. Insira a senha na barra lateral esquerda para carregar o modelo matemático.")
