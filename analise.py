import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



def preprocessar_dados_cursos(dataset):
    # Lista das colunas a serem removidas
    colunas_para_remover = [
        'CO_CINE_ROTULO', 'NO_REGIAO', 'NO_UF', 'SG_UF', 'NO_MUNICIPIO', 'NO_CURSO', 
        'NO_CINE_ROTULO', 'NO_CINE_AREA_GERAL', 'NO_REGIAO_IES', 
        'NO_UF_IES', 'SG_UF_IES', 'NO_MUNICIPIO_IES', 'IN_CAPITAL_IES', 
        'NO_MANTENEDORA', 'NO_IES', 'SG_IES'
    ]
    
    # Remover as colunas especificadas
    dataset = dataset.drop(columns=colunas_para_remover)
    
    # Descartar linhas com valores ausentes
    dataset = dataset.dropna()
    
    return dataset

def preprocessar_dados_ies(dataset):
    # Lista das colunas a serem removidas
    colunas_para_remover = [
        "NO_REGIAO_IES","NO_UF_IES","SG_UF_IES","NO_MUNICIPIO_IES","IN_CAPITAL_IES","NO_MANTENEDORA","NO_IES","SG_IES"
    ]
    
    # Remover as colunas especificadas
    dataset = dataset.drop(columns=colunas_para_remover)
    
    # Descartar linhas com valores ausentes
    dataset = dataset.dropna()
    
    return dataset

def treinar_modelo_regressao(X_train, y_train):
    # Inicializar e treinar diferentes modelos de regressão
    modelos = {
        'Regressão Linear': LinearRegression(),
        'Floresta Aleatória': RandomForestRegressor(),
        'Ridge': Ridge(),
        'Lasso' : Lasso(),
    }
    
    resultados = {}

    for nome, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        resultados[nome] = modelo

    return resultados

def treinar_modelo_clusterizacao(X_train):
    # Inicializar e treinar o modelo de clusterização (KMeans)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_train)
    return kmeans

def avaliar_modelo_regressao(modelo, X_test, y_test):
    resultados = {}

    for nome, mdl in modelo.items():
        y_pred = mdl.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)
        
        resultados[nome] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R²': r2}

    return pd.DataFrame(resultados)

def avaliar_modelo_clusterizacao(modelo, X_test):
    y_pred = modelo.predict(X_test)
    silhouette = silhouette_score(X_test, y_pred)
    homogeneity = homogeneity_score(y_test, y_pred)
    completeness = completeness_score(y_test, y_pred)
    v_measure = v_measure_score(y_test, y_pred)
    adjusted_rand = adjusted_rand_score(y_test, y_pred)
    adjusted_mutual_info = adjusted_mutual_info_score(y_test, y_pred)
    return y_pred, silhouette, homogeneity, completeness, v_measure, adjusted_rand, adjusted_mutual_info

# Carregar o dataset
dataset_cursos = pd.read_csv('C:/airflow-docker/datalake/gold/date=2024-02-16/cursos_concatenated.csv')
dataset_ies = pd.read_csv('C:/airflow-docker/datalake/gold/date=2024-02-16/ies_concatenated.csv')

# Opções para o filtro
opcoes_filtro = ['Cursos', 'IES']

# Adicionar o filtro para escolher a visão
visao_selecionada = st.radio("Escolha a visão:", opcoes_filtro)

# Preprocessar os dados de acordo com a visão selecionada
if visao_selecionada == 'Cursos':
    st.title('Cursos')
    dataset = preprocessar_dados_cursos(dataset_cursos)
else:
    st.title('IES')
    dataset = preprocessar_dados_ies(dataset_ies)

# Dividir os dados em conjunto de treinamento e teste
X = dataset.drop(columns=['NT_GER'])  # Features
y = dataset['NT_GER']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adicionar o filtro para escolher entre Regressão e Clusterização
opcoes_modelo = ['Regressão', 'Clusterização']
modelo_selecionado = st.radio("Escolha o modelo:", opcoes_modelo)

# Treinar o modelo
if modelo_selecionado == 'Regressão':
    modelo = treinar_modelo_regressao(X_train, y_train)
    df_resultados = avaliar_modelo_regressao(modelo, X_test, y_test)
    
    # Exibir gráfico de barras com as métricas
    st.write('## Comparação das Métricas de Regressão')
    fig, ax = plt.subplots()
    sns.barplot(data=df_resultados.T, orient='h', ax=ax)
    ax.set_xlabel('Métrica')
    ax.set_ylabel('Algoritmo')
    ax.set_title('Comparação das Métricas de Regressão')
    st.pyplot(fig)

    # Exibir detalhes das métricas
    st.write('### Detalhes das Métricas')
    st.write(df_resultados)

    # Exibir gráficos de regressão para cada algoritmo
    for nome, mdl in modelo.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pred = mdl.predict(X_test)
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Valores reais')
        ax.set_ylabel('Valores previstos')
        ax.set_title(f'Regressão - {nome}')
        st.pyplot(fig)

elif modelo_selecionado == 'Clusterização':
    modelo = treinar_modelo_clusterizacao(X_train)
    y_pred, silhouette, homogeneity, completeness, v_measure, adjusted_rand, adjusted_mutual_info = avaliar_modelo_clusterizacao(modelo, X_test)
    
    # Exibir gráfico de clusterização
    fig, ax = plt.subplots()
    ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap='viridis')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Clusterização - KMeans')
    st.pyplot(fig)
    
    # Exibir as métricas de clusterização
    st.write('## Métricas de Clusterização')
    st.write(f'Coeficiente Silhouette: {silhouette}')
    st.write(f'Homogeneidade: {homogeneity}')
    st.write(f'Completude: {completeness}')
    st.write(f'V-Measure: {v_measure}')
    st.write(f'Índice Rand Ajustado: {adjusted_rand}')
    st.write(f'Informação Mútua Ajustada: {adjusted_mutual_info}')
