import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Configuração da Página Web do Aplicativo
st.set_page_config(page_title="App Radar IA: Finanças", layout="wide", page_icon="📈")

# =========================================================================
# 🛡️ CONFIGURAÇÃO DE SEGURANÇA (Sua senha de acesso ao App)
SENHA_CORRETA = "MariaVitoria"
# =========================================================================

def enviar_notificacao_telegram(token, chat_id, mensagem):
    if token and chat_id:
        url = f"https://telegram.org{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"}
        try: requests.post(url, json=payload, timeout=5)
        except: pass

# Interface de login na barra lateral
st.sidebar.markdown("# 🔐 Acesso ao App")
senha_digitada = st.sidebar.text_input("Digite sua senha de acesso:", type="password")

if senha_digitada == SENHA_CORRETA:
    st.title("📈 Aplicativo Inteligente de Análise Quantitativa v4.0")
    st.subheader("Filtro Automático de Ativos: Classificação por Risco, Assertividade e Preço")

    # Configurações financeiras na lateral
    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Gestão de Capital")
    capital_total = st.sidebar.number_input("Capital Total para Operar (R$):", value=1000.0, step=100.0)
    risco_max_operacao = st.sidebar.number_input("Risco Máximo por Operação (R$):", value=100.0, step=10.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📢 Notificações Mobile")
    telegram_token = st.sidebar.text_input("Token do Telegram:", type="password", value="")
    telegram_chat_id = st.sidebar.text_input("Chat ID do Telegram:", type="password", value="")

    st.write("### 🔍 Pesquisar outro Ativo na Bolsa")
    ticker_pesquisado = st.text_input("Digite o código de qualquer ação (Ex: SANB4.SA, CPLE6.SA, AAPL):", value="").strip().upper()

    # 1. EXPANSÃO DA LISTA: Monitorando 20 ações principais da bolsa simultaneamente
    LISTA_ACOES = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BBDC4.SA", 
        "MGLU3.SA", "ABEV3.SA", "WEGE3.SA", "ELET3.SA", "RENT3.SA",
        "LREN3.SA", "GGBR4.SA", "ITSA4.SA", "B3SA3.SA", "SUZB3.SA",
        "RAIZ4.SA", "CMIG4.SA", "VVAR3.SA", "JBSS3.SA", "AZUL4.SA"
    ]
    
    if ticker_pesquisado and ticker_pesquisado not in LISTA_ACOES:
        LISTA_ACOES.insert(0, ticker_pesquisado)

    @st.cache_data(ttl=300)
    def processar_aplicativo_ia(lista_ativos, cap, risco_max, tok, cid):
        resultados = []
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

        for ticker in lista_ativos:
            try:
                dados = yf.download(ticker, start="2023-01-01", progress=False, multi_level_index=False, session=session)
                if dados is None or dados.empty or len(dados) < 150: continue
                
                if isinstance(dados.columns, pd.MultiIndex):
                    dados.columns = dados.columns.get_level_values(0)

                dados['Retorno'] = dados['Close'].pct_change()
                dados['Media_Fast'] = dados['Close'].rolling(window=5).mean() / dados['Close']
                dados['Media_Trend'] = dados['Close'].rolling(window=20).mean() / dados['Close']
                dados['Media_Macro'] = dados['Close'].rolling(window=200).mean() / dados['Close']
                dados['Volatilidade'] = dados['Retorno'].rolling(window=10).std()
                dados['Volume_Var'] = dados['Volume'].pct_change()

                dados['Alvo'] = np.where(dados['Close'].shift(-1) > dados['Close'], 1, 0)
                dados = dados.dropna()

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
                probabilidade_alta = float(modelo.predict_proba(ultimos_dados)[0][1] * 100)
                
                stop_loss = preco_atual * 0.97
                preco_venda_alvo = preco_atual * 1.09

                perda_por_acao = preco_atual - stop_loss
                quantidade_lote = int(risco_max / perda_por_acao)
                if (quantidade_lote * preco_atual) > cap:
                    quantidade_lote = int(cap // preco_atual)

                macro_alta = float(dados['Close'].iloc[-1]) > float(dados['Close'].rolling(window=200).mean().iloc[-1])
                
                # Classificação de status
                if probabilidade_alta >= 60.0 and acuracia > 50.0 and macro_alta:
                    veredicto = "🎯 COMPRAR"
                    ordem_prioridade = 1  # Melhores opções no topo
                    status_operacional = "Excelente (Boa para Compra)"
                    
                    msg = f"🚨 *APP SINAL IA* 🚨\nAtivo: {ticker}\nPreço: R$ {preco_atual:.2f}\nQtd: {quantidade_lote}\nAlvo: R$ {preco_venda_alvo:.2f}"
                    enviar_notificacao_telegram(tok, cid, msg)
                else:
                    veredicto = "⏳ FIQUE DE FORA"
                    ordem_prioridade = 2  # Piores opções embaixo
                    quantidade_lote = 0
                    stop_loss = 0.0
                    preco_venda_alvo = 0.0
                    status_operacional = "Aguardar (Não recomendada)"

                resultados.append({
                    "Prioridade": ordem_prioridade,
                    "Ação": ticker,
                    "Preço Atual (R$)": round(preco_atual, 2),
                    "Veredito IA": veredicto,
                    "Status": status_operacional,
                    "Certeza de Alta": round(probabilidade_alta, 1),
                    "Taxa de Acerto (IA)": f"{acuracia:.1f}%",
                    "Qtd Recomendada": quantidade_lote,
                    "Alvo de Venda": round(preco_venda_alvo, 2) if preco_venda_alvo > 0 else 0.0,
                    "Stop Loss": round(stop_loss, 2) if stop_loss > 0 else 0.0
                })
            except: continue
                
        df = pd.DataFrame(resultados)
        
        # 2. SISTEMA DE RANKING: Organiza por melhores decisões (Comprar antes), depois por maior Certeza de Alta e depois por Preço mais barato
        if not df.empty:
            df = df.sort_values(by=["Prioridade", "Certeza de Alta", "Preço Atual (R$)"], ascending=[True, False, True])
            df = df.drop(columns=["Prioridade"]) # Remove coluna de apoio técnico do ranking
        return df

    df_painel = processar_aplicativo_ia(LISTA_ACOES, capital_total, risco_max_operacao, telegram_token, telegram_chat_id)

    col1, col2, col3 = st.columns(3)
    if df_painel is None or df_painel.empty:
        col1.metric("Ações Analisadas", "0")
        col2.metric("Recomendações Ativas", "0")
        st.info("🔄 Sincronizando o aplicativo com a Bolsa de Valores. Aguarde um instante...")
    else:
        ativos_com_sinal = len(df_painel[df_painel["Veredito IA"] == "🎯 COMPRAR"])
        col1.metric("Ações Ativas Monitoradas", len(df_painel))
        col2.metric("Boas para Compra Hoje", ativos_com_sinal)
        col3.metric("Gerenciador de Risco", "Ativo (3% Max Loss)")

        st.write("### 📊 Classificação Geral do Aplicativo (Das Melhores para as Piores / Mais Baratas)")
        
        # Exibe a tabela organizada de forma nativa e limpa
        st.dataframe(df_painel, use_container_width=True)
        
        csv = df_painel.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Exportar Relatório do App para Planilha", data=csv, file_name="app_radar_ia.csv", mime="text/csv")

elif senha_digitada != "":
    st.sidebar.error("❌ Senha Incorreta!")
else:
    st.info("👋 Bem-vindo ao seu App de IA. Insira a chave de acesso na barra lateral esquerda para destravar o painel de investimentos.")
