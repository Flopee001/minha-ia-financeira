import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import requests
import sqlite3
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Configuração Inicial da Página do App
st.set_page_config(page_title="App Radar IA: Finanças", layout="wide", page_icon="📈")

def conectar_banco():
    conn = sqlite3.connect("usuarios_ia.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS usuarios (usuario TEXT PRIMARY KEY, senha TEXT)")
    conn.commit()
    return conn, cursor

def criptografar_senha(senha):
    return hashlib.sha256(senha.encode()).hexdigest()

def cadastrar_usuario(usuario, senha):
    conn, cursor = conectar_banco()
    try:
        cursor.execute("INSERT INTO usuarios (usuario, senha) VALUES (?, ?)", (usuario, criptografar_senha(senha)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def verificar_login(usuario, senha):
    conn, cursor = conectar_banco()
    cursor.execute("SELECT * FROM usuarios WHERE usuario = ? AND senha = ?", (usuario, criptografar_senha(senha)))
    resultado = cursor.fetchone()
    conn.close()
    return resultado is not None

def atualizar_senha(usuario, nova_senha):
    conn, cursor = conectar_banco()
    cursor.execute("UPDATE usuarios SET senha = ? WHERE usuario = ?", (criptografar_senha(nova_senha), usuario))
    conn.commit()
    linhas = cursor.rowcount
    conn.close()
    return linhas > 0

conectar_banco()

st.sidebar.markdown("# 🔐 Painel de Acesso")
aba = st.sidebar.radio("Escolha uma opção:", ["Entrar no Sistema", "Criar Nova Conta", "Redefinir Minha Senha"])

login_confirmado = False
usuario_ativo = ""

if aba == "Entrar no Sistema":
    st.sidebar.subheader("Login")
    user_input = st.sidebar.text_input("Nome de Usuário:")
    pass_input = st.sidebar.text_input("Senha de Acesso:", type="password")
    if st.sidebar.button("Entrar"):
        if verificar_login(user_input, pass_input):
            st.session_state["login_confirmado"] = True
            st.session_state["usuario_ativo"] = user_input
            st.sidebar.success("🔓 Acesso liberado!")
            st.rerun()
        else:
            st.sidebar.error("❌ Usuário ou senha incorretos.")

elif aba == "Criar Nova Conta":
    st.sidebar.subheader("Cadastro")
    novo_user = st.sidebar.text_input("Escolha um Nome de Usuário:")
    novo_pass = st.sidebar.text_input("Escolha uma Senha:", type="password")
    confirmar_pass = st.sidebar.text_input("Confirme a Senha:", type="password")
    if st.sidebar.button("Salvar Cadastro"):
        if novo_pass != confirmar_pass:
            st.sidebar.error("❌ As senhas não batem.")
        elif novo_user and novo_pass:
            if cadastrar_usuario(novo_user, novo_pass):
                st.sidebar.success("🎉 Conta criada! Faça o login na outra aba.")
            else:
                st.sidebar.error("❌ Usuário já existe.")

elif aba == "Redefinir Minha Senha":
    st.sidebar.subheader("Alterar Senha")
    user_redefinir = st.sidebar.text_input("Usuário:")
    senha_atual = st.sidebar.text_input("Senha Atual:", type="password")
    nova_senha = st.sidebar.text_input("Nova Senha:", type="password")
    if st.sidebar.button("Atualizar Senha"):
        if verificar_login(user_redefinir, senha_atual):
            if atualizar_senha(user_redefinir, nova_senha):
                st.sidebar.success("✅ Senha alterada!")
        else:
            st.sidebar.error("❌ Dados inválidos.")

if "login_confirmado" in st.session_state and st.session_state["login_confirmado"]:
    login_confirmado = True
    usuario_ativo = st.session_state["usuario_ativo"]

if login_confirmado:
    st.title("📈 Aplicativo Inteligente de Análise Quantitativa v4.5")
    st.subheader(f"Painel Ativo — Usuário: {usuario_ativo}")

    st.sidebar.markdown("---")
    capital_total = st.sidebar.number_input("Capital Total (R$):", value=1000.0)
    risco_max_operacao = st.sidebar.number_input("Risco Máximo (R$):", value=100.0)

    telegram_token = st.sidebar.text_input("Token Telegram:", type="password")
    telegram_chat_id = st.sidebar.text_input("Chat ID:", type="password")

    st.write("### 🔍 Pesquisar uma Ação Extra")
    ticker_pesquisado = st.text_input("Digite o código de uma nova empresa (Ex: SANB4.SA, CPLE6.SA) para embutir na lista fixa:", value="").strip().upper()

    # 1. LISTA FIXA GARANTIDA: Sempre rodará estes papéis principais
    LISTA_ACOES = ["PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BBDC4.SA", "MGLU3.SA", "ABEV3.SA", "WEGE3.SA", "ELET3.SA", "RENT3.SA"]
    
    # Se você pesquisar algo, o app insere esse ativo temporariamente junto com as fixas
    if ticker_pesquisado and ticker_pesquisado not in LISTA_ACOES:
        LISTA_ACOES.insert(0, ticker_pesquisado)

    @st.cache_data(ttl=300)
    def processar_ia(lista_ativos, cap, risco_max):
        resultados = []
        session = requests.Session()
        session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        
        for ticker in lista_ativos:
            try:
                # Sincronização direta e estável usando os cabeçalhos corretos para impedir bloqueios da API
                dados = yf.download(ticker, start="2023-01-01", progress=False, multi_level_index=False, session=session)
                if dados is None or dados.empty or len(dados) < 100: 
                    continue
                
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

                X = dados[['Media_Fast', 'Media_Trend', 'Media_Macro', 'Volatilidade', 'Retorno', 'Volume_Var']]
                y = dados['Alvo']
                
                div = int(len(X) * 0.8)
                modelo = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
                modelo.fit(X.iloc[:div], y.iloc[:div])

                acuracia = accuracy_score(y.iloc[div:], modelo.predict(X.iloc[div:])) * 100
                preco_atual = float(dados['Close'].iloc[-1])
                prob_alta = float(modelo.predict_proba(X.tail(1)) * 100)

                macro_alta = preco_atual > float(dados['Close'].rolling(window=200).mean().iloc[-1])

                if prob_alta >= 60.0 and acuracia > 50.0 and macro_alta:
                    veredicto, prioridade, status = "🎯 COMPRAR", 1, "Excelente"
                    stop_loss, preco_alvo = preco_atual * 0.97, preco_atual * 1.09
                    dif = preco_atual - stop_loss
                    qtd = int(risco_max / dif) if dif > 0 else 0
                    if (qtd * preco_atual) > cap: 
                        qtd = int(cap // preco_atual)
                else:
                    veredicto, prioridade, status = "⏳ FIQUE DE FORA", 2, "Aguardar"
                    qtd, stop_loss, preco_alvo = 0, 0.0, 0.0

                resultados.append({
                    "Prioridade": prioridade, "Ação": ticker, "Preço Atual (R$)": round(preco_atual, 2),
                    "Veredito IA": veredicto, "Status": status, "Certeza de Alta": round(prob_alta, 1),
                    "Taxa de Acerto": f"{acuracia:.1f}%", "Qtd Recomendada": qtd,
                    "Alvo de Venda": round(preco_alvo, 2), "Stop Loss": round(stop_loss, 2)
                })
            except:
                continue
        
        df = pd.DataFrame(resultados)
        if not df.empty:
            df = df.sort_values(by=["Prioridade", "Certeza de Alta"], ascending=[True, False]).drop(columns=["Prioridade"])
        return df

    # GATILHO COMPLETO: Roda o processamento diretamente para todas as ações automaticamente
    df_painel = processar_ia(LISTA_ACOES, capital_total, risco_max_operacao)

    if not df_painel.empty:
        col1, col2, col3 = st.columns(3)
        ativos_com_sinal = len(df_painel[df_painel["Veredito IA"] == "🎯 COMPRAR"])
        col1.metric("Ações Ativas Monitoradas", len(df_painel))
        col2.metric("Boas para Compra Hoje", ativos_com_sinal)
        col3.metric("Gerenciador de Risco", "Ativo (3% Max Loss)")

        st.write("### 📊 Ranking das Melhores Ações para Operar")
        st.dataframe(df_painel, use_container_width=True)
        
        if telegram_token and telegram_chat_id:
            for _, row in df_painel.iterrows():
                if row["Veredito IA"] == "🎯 COMPRAR":
                    url = f"https://telegram.org{telegram_token}/sendMessage"
                    msg = f"🚨 *COMPRA IA* 🚨\nAtivo: {row['Ação']}\nPreço: R$ {row['Preço Atual (R$)']}\nQtd: {row['Qtd Recomendada']}"
                    try: requests.post(url, json={"chat_id": telegram_chat_id, "text": msg, "parse_mode": "Markdown"})
                    except: pass
    else:
        st.info("🔄 Sincronizando o aplicativo de IA com a Bolsa de Valores. Aguarde um instante para a montagem automática das linhas...")

else:
    st.title("📈 Aplicativo Inteligente de Análise Quantitativa v4.5")
    st.info("🔒 Sistema Bloqueado. Crie uma conta ou faça login na barra lateral para ver o ranking automático da IA.")
