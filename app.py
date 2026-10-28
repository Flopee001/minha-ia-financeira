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

# =========================================================================
# 🗄️ GERENCIAMENTO DO BANCO DE DADOS (SQLite)
# =========================================================================
def conectar_banco():
    conn = sqlite3.connect("usuarios_ia.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            usuario TEXT PRIMARY KEY,
            senha TEXT
        )
    """)
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
    linhas_afetadas = cursor.rowcount
    conn.close()
    return linhas_afetadas > 0

# Inicializa a tabela do banco de dados na inicialização
conectar_banco()

# =========================================================================
# 🔑 INTERFACE DE ACESSO (Login, Cadastro e Redefinição)
# =========================================================================
st.sidebar.markdown("# 🔐 Painel de Acesso")
aba_autenticacao = st.sidebar.radio("Escolha uma opção:", ["Entrar no Sistema", "Criar Nova Conta", "Redefinir Minha Senha"])

login_confirmado = False
usuario_ativo = ""

if aba_autenticacao == "Entrar no Sistema":
    st.sidebar.subheader("Login")
    user_input = st.sidebar.text_input("Nome de Usuário:")
    pass_input = st.sidebar.text_input("Senha de Acesso:", type="password")
    
    if st.sidebar.button("Entrar"):
        if verificar_login(user_input, pass_input):
            st.session_state["login_confirmado"] = True
            st.session_state["usuario_ativo"] = user_input
            st.sidebar.success(f"🔓 Acesso liberado! Bem-vindo, {user_input}.")
        else:
            st.sidebar.error("❌ Usuário ou senha incorretos.")

elif aba_autenticacao == "Criar Nova Conta":
    st.sidebar.subheader("Cadastro de Novo Usuário")
    novo_user = st.sidebar.text_input("Escolha um Nome de Usuário:")
    novo_pass = st.sidebar.text_input("Escolha uma Senha Segura:", type="password")
    confirmar_pass = st.sidebar.text_input("Confirme a Senha:", type="password")
    
    if st.sidebar.button("Salvar Cadastro"):
        if not novo_user or not novo_pass:
            st.sidebar.warning("⚠️ Preencha todos os campos obrigatórios.")
        elif novo_pass != confirmar_pass:
            st.sidebar.error("❌ As senhas digitadas não são iguais.")
        else:
            sucesso = cadastrar_usuario(novo_user, novo_pass)
            if sucesso:
                st.sidebar.success("🎉 Conta criada com sucesso! Mude para a aba 'Entrar no Sistema' para fazer login.")
            else:
                st.sidebar.error("❌ Este nome de usuário já está sendo usado.")

elif aba_autenticacao == "Redefinir Minha Senha":
    st.sidebar.subheader("Alterar Senha Existente")
    user_redefinir = st.sidebar.text_input("Confirme seu Nome de Usuário:")
    senha_atual_verificar = st.sidebar.text_input("Digite sua Senha Atual:", type="password")
    nova_senha_input = st.sidebar.text_input("Digite a NOVA Senha:", type="password")
    
    if st.sidebar.button("Atualizar Senha"):
        if verificar_login(user_redefinir, senha_atual_verificar):
            if atualizar_senha(user_redefinir, nova_senha_input):
                st.sidebar.success("✅ Senha redefinida com sucesso! Faça login com a nova credencial.")
            else:
                st.sidebar.error("❌ Erro interno ao atualizar. Tente novamente.")
        else:
            st.sidebar.error("❌ Usuário ou senha atual incorretos. Não foi possível validar a alteração.")

# Mantém o estado do login ativo enquanto navega no app
if "login_confirmado" in st.session_state and st.session_state["login_confirmado"]:
    login_confirmado = True
    usuario_ativo = st.session_state["usuario_ativo"]

# =========================================================================
# 🤖 INTEGRAÇÃO DO CORE DA IA (Carrega apenas após login válido)
# =========================================================================
if login_confirmado:
    st.title("📈 Aplicativo Inteligente de Análise Quantitativa v4.5")
    st.subheader(f"Painel Operacional Ativo — Analisando como: {usuario_ativo}")

    st.sidebar.markdown("---")
    st.sidebar.subheader("💰 Gestão de Capital")
    capital_total = st.sidebar.number_input("Capital Total para Operar (R$):", value=1000.0, step=100.0)
    risco_max_operacao = st.sidebar.number_input("Risco Máximo por Operação (R$):", value=100.0, step=10.0)

    def enviar_notificacao_telegram(token, chat_id, mensagem):
        if token and chat_id:
            url = f"https://telegram.org{token}/sendMessage"
            payload = {"chat_id": chat_id, "text": mensagem, "parse_mode": "Markdown"}
            try: requests.post(url, json=payload, timeout=5)
            except: pass

    st.sidebar.markdown("---")
    st.sidebar.subheader("📢 Notificações Mobile")
    telegram_token = st.sidebar.text_input("Token do Telegram:", type="password", value="")
    telegram_chat_id = st.sidebar.text_input("Chat ID do Telegram:", type="password", value="")

    st.write("### 🔍 Pesquisar outro Ativo na Bolsa")
    ticker_pesquisado = st.text_input("Digite o código de qualquer ação (Ex: SANB4.SA, CPLE6.SA, AAPL):", value="").strip().upper()

    LISTA_ACOES = [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBAS3.SA", "BBDC4.SA", 
        "MGLU3.SA", "ABEV3.SA", "WEGE3.SA", "ELET3.SA", "RENT3.SA",
        "LREN3.SA", "GGBR4.SA", "ITSA4.SA", "B3SA3.SA", "SUZB3.SA",
        "RAIZ4.SA", "CMIG4.SA", "JBSS3.SA", "AZUL4.SA"
    ]
    
    if ticker_pesquisado and ticker_pesquisado not in LISTA_ACOES:
        LISTA_ACOES.insert(0, ticker_pesquisado)

    @st.cache_data(ttl=300)
    def processar_aplicativo_ia(lista_ativos, cap, risco_max):
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
                probabilidade_alta = float(modelo.predict_proba(ultimos_dados) * 100)
                
                stop_loss = preco_atual * 0.97
                preco_venda_alvo = preco_atual * 1.09

                perda_por_acao = preco_atual - stop_loss
                quantidade_lote = int(risco_max / perda_por_acao)
                if (quantidade_lote * preco_atual) > cap:
                    quantidade_lote = int(cap // preco_atual)

                macro_alta = float(dados['Close'].iloc[-1]) > float(dados['Close'].rolling(window=200).mean().iloc[-1])
                
                if probabilidade_alta >= 60.0 and acuracia > 50.0 and macro_alta:
                    veredicto = "🎯 COMPRAR"
                    ordem_prioridade = 1
                    status_operacional = "Excelente (Boa para Compra)"
                else:
                    veredicto = "⏳ FIQUE DE FORA"
                    ordem_prioridade = 2
                    quantidade_lote = 0
                    stop_loss = 0.0
                    preco_venda_alvo = 0.0
