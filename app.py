import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Radar de IA: Mercado Financeiro", layout="wide")
st.title("🤖 Sistema Quantitativo de IA & Gestão de Risco v2.0")
st.subheader("Monitoramento automatizado de ativos e mitigação de perdas")

# Lista com as principais ações da Bolsa para monitoramento
LISTA_ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BBDC4.SA", "MGLU3.SA", "ABEV3.SA", "WEGE3.SA"]

@st.cache_data(ttl=3600)  # Atualiza os dados de hora em hora automaticamente
def processar_ia_mercado():
    resultados = []
    for ticker in LISTA_ACOES:
        try:
            # Coleta os dados automáticos até o momento atual
            dados = yf.download(ticker, start="2023-01-01", progress=False)
            if dados.empty: continue
            dados.columns = dados.columns.get_level_values(0)

            # Cálculo das variáveis para o aprendizado da IA
            dados['Retorno'] = dados['Close'].pct_change()
            dados['Media_5'] = dados['Close'].rolling(window=5).mean() / dados['Close']
            dados['Media_20'] = dados['Close'].rolling(window=20).mean() / dados['Close']
            dados['Volatilidade_10'] = dados['Retorno'].rolling(window=10).std()
            dados['Volume_Var'] = dados['Volume'].pct_change()

            # Alvo matemático (Se vai subir ou cair)
            dados['Alvo'] = np.where(dados['Close'].shift(-1) > dados['Close'], 1, 0)
            dados = dados.dropna()

            recursos = ['Media_5', 'Media_20', 'Volatilidade_10', 'Retorno', 'Volume_Var']
            X = dados[recursos]
            y = dados['Alvo']

            # Divisão dos dados para teste de acerto
            divisao = int(len(X) * 0.8)
            X_treino, X_teste = X.iloc[:divisao], X.iloc[divisao:]
            y_treino, y_teste = y.iloc[:divisao], y.iloc[divisao:]

            # Treinamento da Inteligência Artificial
            modelo = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=10)
            modelo.fit(X_treino, y_treino)

            # Teste de assertividade histórica
            previsoes_teste = modelo.predict(X_teste)
            acuracia = accuracy_score(y_teste, previsoes_teste) * 100

            # Previsão para o pregão atual
            ultimos_dados = X.tail(1)
            preco_atual = float(dados['Close'].iloc[-1])
            
            probabilidades = modelo.predict_proba(ultimos_dados)
            probabilidade_alta = float(probabilidades[0][1] * 100)
            
            # Localiza o preço mais barato recente (Mínima de 5 dias)
            suporte_preco_barato = float(dados['Low'].tail(5).min())

            # Filtros de Segurança Ativos para Mitigação de Perdas
            if probabilidade_alta >= 60.0 and acuracia > 50.0:
                veredicto = "🎯 COMPRAR"
                stop_loss = preco_atual * 0.97  # Limita prejuízo máximo em 3%
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
        except:
            continue
            
    return pd.DataFrame(resultados)

# Montagem do Painel Visual
df_painel = processar_ia_mercado()

col1, col2, col3 = st.columns(3)
oportunidades_hoje = len(df_painel[df_painel["Veredito do Sistema"] == "🎯 COMPRAR"])
col1.metric("Ações Monitoradas", len(LISTA_ACOES))
col2.metric("Oportunidades de Compra", oportunidades_hoje)
col3.metric("Filtro de Risco", "3% Máx Loss")

st.write("### 📋 Painel de Tomada de Decisão Real")
st.dataframe(df_painel, use_container_width=True)

# Geração automática da planilha para download
csv = df_painel.to_csv(index=False).encode('utf-8')
st.download_button(label="📥 Baixar Relatório em Planilha", data=csv, file_name="radar_ia.csv", mime="text/csv")
