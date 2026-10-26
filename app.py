import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Configuração da Página Web
st.set_page_config(page_title="Radar de IA: Mercado Financeiro", layout="wide")

# =========================================================================
# 🛡️ SISTEMA DE SEGURANÇA: DIGITE SUA SENHA EXCLUSIVA AQUI
SENHA_CORRETA = "MariaVitoria"
# =========================================================================

# Criar a interface de login visual na barra lateral
st.sidebar.title("🔐 Área Restrita")
senha_digitada = st.sidebar.text_input("Digite a senha para acessar a IA:", type="password")

# Se a senha estiver correta, o site abre o robô
if senha_digitada == SENHA_CORRETA:
    st.title("🤖 Sistema Quantitativo de IA & Gestão de Risco v2.0")
    st.subheader("Monitoramento automatizado de ativos e mitigação de perdas")

    LISTA_ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BBDC4.SA", "MGLU3.SA", "ABEV3.SA", "WEGE3.SA"]

    @st.cache_data(ttl=3600)
    def processar_ia_mercado():
        resultados = []
        for ticker in LISTA_ACOES:
            try:
                # Mudança estratégica: Baixa os dados sem travar data final rígida
                dados = yf.download(ticker, start="2023-01-01", progress=False)
                if dados is None or dados.empty or len(dados) < 30: 
                    continue
                
                dados.columns = dados.columns.get_level_values(0)

                dados['Retorno'] = dados['Close'].pct_change()
                dados['Media_5'] = dados['Close'].rolling(window=5).mean() / dados['Close']
                dados['Media_20'] = dados['Close'].rolling(window=20).mean() / dados['Close']
                dados['Volatilidade_10'] = dados['Retorno'].rolling(window=10).std()
                dados['Volume_Var'] = dados['Volume'].pct_change()

                dados['Alvo'] = np.where(dados['Close'].shift(-1) > dados['Close'], 1, 0)
                dados = dados.dropna()

                if len(dados) < 10:
                    continue

                recursos = ['Media_5', 'Media_20', 'Volatilidade_10', 'Retorno', 'Volume_Var']
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
                probabilidade_alta = float(probabilidades[0][1] * 100)
                
                suporte_preco_barato = float(dados['Low'].tail(5).min())

                if probabilidade_alta >= 60.0 and acuracia > 50.0:
                    veredicto = "🎯 COMPRAR"
                    stop_loss = preco_atual * 0.97
                    acao_recomendada = "Configurar Ordem de Compra"
                else:
                    veredicto = "⏳ FIQUE DE FORA"
                    stop_loss = 0.0
                    acao_recomendada = "Aguardar Oportunidade"

                resultados.append({
                    "Ação": ticker,
                    "Preço Atual": f"R$ {preco_atual:.2f}",
                    "Preço Mais Barato (5d)": f"R$ {suporte_preco_barato:.2f}",
                    "Taxa de Acerto (IA)": f"{acuracia:.1f}%",
                    "Certeza de Alta": f"{probabilidade_alta:.1f}%",
                    "Veredito do Sistema": veredicto,
                    "Proteção Stop Loss": f"R$ {stop_loss:.2f}" if stop_loss > 0 else "N/A",
                    "Ação Estratégica": acao_recomendada
                })
            except Exception as e:
                continue
                
        return pd.DataFrame(resultados)

    df_painel = processar_ia_mercado()

    # 🛡️ TRAVA DE SEGURANÇA: Se a tabela estiver vazia por falta de conexão, o app não quebra
    if df_painel.empty:
        st.warning("⚠️ Conexão temporária com a API financeira falhou. Aguarde alguns segundos e atualize a página.")
    else:
        col1, col2, col3 = st.columns(3)
        oportunidades_hoje = len(df_painel[df_painel["Veredito do Sistema"] == "🎯 COMPRAR"])
        col1.metric("Ações Monitoradas", len(LISTA_ACOES))
        col2.metric("Oportunidades de Compra", oportunidades_hoje)
        col3.metric("Filtro de Risco", "3% Máx Loss")

        st.write("### 📋 Painel de Tomada de Decisão Real")
        st.dataframe(df_painel, use_container_width=True)

        csv = df_painel.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Baixar Relatório em Planilha", data=csv, file_name="radar_ia.csv", mime="text/csv")

# Se o usuário errar a senha
elif senha_digitada != "":
    st.sidebar.error("❌ Senha Incorreta! Acesso negado.")
    st.info("Por favor, insira a credencial correta na barra lateral para carregar a Inteligência Artificial.")
else:
    st.info("🔒 Sistema Bloqueado. Insira a senha na barra lateral esquerda para visualizar os dados de inteligência.")
