"""
run_analysis.py
================
Pipeline completo de segmentacao (clustering) de clientes de cartao de credito.

Reproduz integralmente os experimentos reportados no artigo:
  "Segmentacao de Clientes de Cartao de Credito por Meio de Tecnicas de
  Agrupamento Nao Supervisionado: uma Analise Comparativa entre K-Means e
  Clusterizacao Hierarquica Aglomerativa"

Etapas executadas (ver README.md para detalhes):
  1. Carregamento e limpeza da base UCI "Default of Credit Card Clients";
  2. Engenharia de atributos (media de fatura/pagamento, taxa de utilizacao,
     quantidade de atrasos, faixa etaria, perfil de risco heuristico);
  3. Padronizacao (z-score) das variaveis usadas no agrupamento;
  4. Determinacao do numero ideal de clusters (k) via Metodo do Cotovelo e
     Coeficiente de Silhueta, k = 2..10;
  5. Treinamento do K-Means final (k=4) sobre a base completa;
  6. Treinamento da Clusterizacao Hierarquica Aglomerativa (linkage='ward')
     sobre uma amostra de 5.000 registros (restricao de memoria O(n^2));
  7. Calculo de metricas de validacao interna (Silhueta, Davies-Bouldin,
     Calinski-Harabasz) e do Indice de Rand Ajustado (ARI) entre os metodos;
  8. Geracao de todas as figuras e tabelas usadas no artigo.

Uso:
    python3 run_analysis.py

Requisitos: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import json
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)
warnings.filterwarnings('ignore')

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['savefig.bbox'] = 'tight'

# Diretorio deste script: entradas/saidas ficam ao lado dele por padrao,
# tornando o pipeline portavel para qualquer maquina.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, 'UCI_Credit_Card.csv')
OUT = os.path.join(BASE_DIR, 'resultados')
FIGS_OUT = os.path.join(BASE_DIR, 'figuras')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS_OUT, exist_ok=True)

results = {}

# ---------------------------------------------------------------
# 1. Carregamento da base bruta
# ---------------------------------------------------------------
df_raw = pd.read_csv(INPUT_CSV)
results['n_raw'] = df_raw.shape[0]
results['p_raw'] = df_raw.shape[1]
print(f"Dataset carregado: {df_raw.shape[0]:,} registros x {df_raw.shape[1]} variaveis")

# ---------------------------------------------------------------
# 2. Renomeacao de colunas e limpeza basica
# ---------------------------------------------------------------
df = df_raw.copy()
mapa_colunas = {
    'ID': 'id', 'LIMIT_BAL': 'limite_credito',
    'SEX': 'sexo', 'EDUCATION': 'escolaridade', 'MARRIAGE': 'estado_civil',
    'AGE': 'idade',
    'PAY_0': 'status_pag_set', 'PAY_2': 'status_pag_ago',
    'PAY_3': 'status_pag_jul', 'PAY_4': 'status_pag_jun',
    'PAY_5': 'status_pag_mai', 'PAY_6': 'status_pag_abr',
    'BILL_AMT1': 'fatura_set', 'BILL_AMT2': 'fatura_ago',
    'BILL_AMT3': 'fatura_jul', 'BILL_AMT4': 'fatura_jun',
    'BILL_AMT5': 'fatura_mai', 'BILL_AMT6': 'fatura_abr',
    'PAY_AMT1': 'pagamento_set', 'PAY_AMT2': 'pagamento_ago',
    'PAY_AMT3': 'pagamento_jul', 'PAY_AMT4': 'pagamento_jun',
    'PAY_AMT5': 'pagamento_mai', 'PAY_AMT6': 'pagamento_abr',
    'default.payment.next.month': 'inadimplente',
}
df.rename(columns=mapa_colunas, inplace=True)

df.drop(columns=['id'], inplace=True)
n_before_dedup = df.shape[0]
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
results['n_duplicates_removed'] = n_before_dedup - df.shape[0]

df['sexo']         = df['sexo'].map({1: 'Masculino', 2: 'Feminino'})
df['escolaridade'] = df['escolaridade'].map(
    {0:'Outros', 1:'Pós-graduação', 2:'Graduação', 3:'Ensino Médio',
     4:'Outros', 5:'Outros', 6:'Outros'})
df['estado_civil'] = df['estado_civil'].map(
    {0:'Outros', 1:'Casado', 2:'Solteiro', 3:'Outros'})
df['inadimplente'] = df['inadimplente'].astype(int)

cols_fatura    = ['fatura_set','fatura_ago','fatura_jul','fatura_jun','fatura_mai','fatura_abr']
cols_pagamento = ['pagamento_set','pagamento_ago','pagamento_jul','pagamento_jun','pagamento_mai','pagamento_abr']
cols_status    = ['status_pag_set','status_pag_ago','status_pag_jul','status_pag_jun','status_pag_mai','status_pag_abr']

df['media_fatura']    = df[cols_fatura].mean(axis=1).round(2)
df['media_pagamento'] = df[cols_pagamento].mean(axis=1).round(2)
df['taxa_utilizacao'] = np.clip(
    np.where(df['limite_credito'] > 0, df['media_fatura'] / df['limite_credito'], 0), 0, 1.0
).round(4)
df['qtd_atrasos'] = (df[cols_status] >= 1).sum(axis=1)

bins, labels = [0,25,35,45,55,100], ['Até 25','26–35','36–45','46–55','Acima de 55']
df['faixa_etaria'] = pd.cut(df['idade'], bins=bins, labels=labels, right=True)

def classificar_risco(row):
    if row['taxa_utilizacao'] >= 0.8 or row['qtd_atrasos'] >= 3:
        return 'Alto'
    elif row['taxa_utilizacao'] >= 0.5 or row['qtd_atrasos'] >= 1:
        return 'Médio'
    return 'Baixo'
df['perfil_risco'] = df.apply(classificar_risco, axis=1)

results['n_clean'] = df.shape[0]
print(f"Pre-processamento concluido! Shape: {df.shape}")

desc_stats = df[['limite_credito','idade','media_fatura','media_pagamento','taxa_utilizacao','qtd_atrasos']].describe().round(2)
desc_stats.to_csv(f'{OUT}/estatisticas_descritivas.csv')
results['taxa_inadimplencia_geral'] = round(df['inadimplente'].mean()*100, 2)

# 3. Selecao de variaveis e padronizacao
features = ['limite_credito','idade','media_fatura','media_pagamento','taxa_utilizacao','qtd_atrasos']
X = df[features].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=features)
X_scaled_df.describe().round(4).to_csv(f'{OUT}/estatisticas_padronizadas.csv')

# 4. Determinacao de k
K_RANGE = range(2, 11)
inercias, silhuetas, db_scores, ch_scores = [], [], [], []
for k in K_RANGE:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels_k = km.fit_predict(X_scaled)
    inercias.append(km.inertia_)
    silhuetas.append(silhouette_score(X_scaled, labels_k))
    db_scores.append(davies_bouldin_score(X_scaled, labels_k))
    ch_scores.append(calinski_harabasz_score(X_scaled, labels_k))

k_table = pd.DataFrame({'k': list(K_RANGE), 'inercia': inercias, 'silhueta': silhuetas,
                         'davies_bouldin': db_scores, 'calinski_harabasz': ch_scores})
k_table.to_csv(f'{OUT}/tabela_k.csv', index=False)
print(k_table.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ks = list(K_RANGE)
axes[0].plot(ks, inercias, 'o-', color='#4C9BE8', linewidth=2, markersize=7)
axes[0].set_title('Método do Cotovelo')
axes[0].set_xlabel('Número de Clusters (k)')
axes[0].set_ylabel('Inércia')
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
for x, y in zip(ks, inercias):
    axes[0].annotate(f'{y:,.0f}', (x, y), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=7)

axes[1].plot(ks, silhuetas, 's-', color='#E8634C', linewidth=2, markersize=7)
axes[1].set_title('Coeficiente de Silhueta')
axes[1].set_xlabel('Número de Clusters (k)')
axes[1].set_ylabel('Silhueta Média')
for x, y in zip(ks, silhuetas):
    axes[1].annotate(f'{y:.3f}', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)

plt.suptitle('Determinação do Número Ideal de Clusters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig1_cotovelo_silhueta.png', dpi=150)
plt.close()

k_otimo = int(k_table.loc[k_table['silhueta'].idxmax(), 'k'])
results['k_otimo_por_silhueta'] = k_otimo
results['silhueta_k_otimo'] = float(k_table.loc[k_table['k']==k_otimo, 'silhueta'].values[0])
print(f"k otimo pela silhueta: {k_otimo}")

K_OTIMO = 4

# 5. K-Means final
km_final = KMeans(n_clusters=K_OTIMO, init='k-means++', n_init=20, random_state=42)
df['cluster_kmeans'] = km_final.fit_predict(X_scaled)

sil_final = silhouette_score(X_scaled, df['cluster_kmeans'])
db_final = davies_bouldin_score(X_scaled, df['cluster_kmeans'])
ch_final = calinski_harabasz_score(X_scaled, df['cluster_kmeans'])

results['kmeans_k'] = K_OTIMO
results['kmeans_inertia'] = float(km_final.inertia_)
results['kmeans_silhueta'] = float(sil_final)
results['kmeans_davies_bouldin'] = float(db_final)
results['kmeans_calinski_harabasz'] = float(ch_final)

dist_kmeans = df['cluster_kmeans'].value_counts().sort_index()
results['kmeans_distribuicao'] = dist_kmeans.to_dict()
print("K-Means:", results['kmeans_silhueta'], dist_kmeans.to_dict())

# 6. Agglomerative Clustering
# OBS: a clusterizacao hierarquica aglomerativa tem complexidade de memoria O(n^2),
# o que inviabiliza sua aplicacao direta sobre os ~30 mil registros no ambiente
# computacional disponivel (~4 GB RAM). Segue-se a pratica usual na literatura de
# aplicar o metodo hierarquico sobre uma amostra aleatoria estratificada e comparar
# os resultados com o K-Means treinado sobre a base completa.
N_AMOSTRA_AGG = 5000
np.random.seed(42)
idx_agg = np.random.choice(X_scaled.shape[0], size=N_AMOSTRA_AGG, replace=False)
X_agg_sample = X_scaled[idx_agg]

agg = AgglomerativeClustering(n_clusters=K_OTIMO, linkage='ward')
labels_agg_sample = agg.fit_predict(X_agg_sample)

sil_agg = silhouette_score(X_agg_sample, labels_agg_sample)
db_agg = davies_bouldin_score(X_agg_sample, labels_agg_sample)
ch_agg = calinski_harabasz_score(X_agg_sample, labels_agg_sample)

results['agg_n_amostra'] = N_AMOSTRA_AGG
results['agg_k'] = K_OTIMO
results['agg_silhueta'] = float(sil_agg)
results['agg_davies_bouldin'] = float(db_agg)
results['agg_calinski_harabasz'] = float(ch_agg)

dist_agg = pd.Series(labels_agg_sample).value_counts().sort_index()
results['agg_distribuicao'] = dist_agg.to_dict()
print("Agglomerative (amostra):", results['agg_silhueta'], dist_agg.to_dict())

labels_kmeans_sample = df['cluster_kmeans'].values[idx_agg]
ari = adjusted_rand_score(labels_kmeans_sample, labels_agg_sample)
results['ari_kmeans_vs_agg'] = float(ari)
print("ARI (amostra):", ari)

df['cluster_hierarquico'] = np.nan
df.loc[df.index[idx_agg], 'cluster_hierarquico'] = labels_agg_sample

# 7. Dendrograma (amostra)
np.random.seed(42)
sample_idx = np.random.choice(X_scaled.shape[0], size=300, replace=False)
X_sample = X_scaled[sample_idx]
Z = linkage(X_sample, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=8., show_contracted=True)
plt.title('Dendrograma – Clusterização Hierárquica (amostra n=300)')
plt.xlabel('Clientes (agrupados)')
plt.ylabel('Distância (Ward)')
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig2_dendrograma.png', dpi=150)
plt.close()

# 8. PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
results['pca_var_explicada'] = pca.explained_variance_ratio_.tolist()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
scatter1 = axes[0].scatter(X_pca[:,0], X_pca[:,1], c=df['cluster_kmeans'], cmap='viridis', s=8, alpha=0.6)
axes[0].set_title(f'K-Means (k={K_OTIMO}) – Projeção PCA')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

X_pca_agg_sample = pca.transform(X_agg_sample)
scatter2 = axes[1].scatter(X_pca_agg_sample[:,0], X_pca_agg_sample[:,1], c=labels_agg_sample, cmap='viridis', s=8, alpha=0.6)
axes[1].set_title(f'Agglomerative (k={K_OTIMO}, amostra n={N_AMOSTRA_AGG}) – Projeção PCA')
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.suptitle('Visualização dos Agrupamentos via PCA', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig3_pca_clusters.png', dpi=150)
plt.close()

# 9. Silhueta detalhada
sample_sizes = 3000
np.random.seed(42)
sil_sample_idx = np.random.choice(X_scaled.shape[0], size=sample_sizes, replace=False)
X_sil = X_scaled[sil_sample_idx]
labels_sil = df['cluster_kmeans'].values[sil_sample_idx]
sil_values = silhouette_samples(X_sil, labels_sil)

fig, ax = plt.subplots(figsize=(9,6))
y_lower = 10
colors = sns.color_palette('viridis', K_OTIMO)
for i in range(K_OTIMO):
    ith_vals = sil_values[labels_sil == i]
    ith_vals.sort()
    size_i = ith_vals.shape[0]
    y_upper = y_lower + size_i
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, facecolor=colors[i], edgecolor=colors[i], alpha=0.8)
    ax.text(-0.05, y_lower + 0.5*size_i, str(i))
    y_lower = y_upper + 10
ax.axvline(x=sil_final, color='red', linestyle='--', label=f'Média = {sil_final:.3f}')
ax.set_title('Análise de Silhueta por Cluster – K-Means (amostra n=3000)')
ax.set_xlabel('Coeficiente de Silhueta')
ax.set_ylabel('Cluster')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig4_silhueta_detalhada.png', dpi=150)
plt.close()

# 10. Perfil dos clusters
perfil = df.groupby('cluster_kmeans')[features].mean().round(2)
perfil['n_clientes'] = df.groupby('cluster_kmeans').size()
perfil['pct_clientes'] = (perfil['n_clientes'] / perfil['n_clientes'].sum() * 100).round(1)
perfil['taxa_inadimplencia_%'] = (df.groupby('cluster_kmeans')['inadimplente'].mean()*100).round(2)
perfil.to_csv(f'{OUT}/perfil_clusters_kmeans.csv')
print(perfil)

results['perfil_clusters'] = perfil.reset_index().to_dict(orient='records')

risco_cluster = pd.crosstab(df['cluster_kmeans'], df['perfil_risco'], normalize='index').round(3)*100
risco_cluster.to_csv(f'{OUT}/risco_por_cluster.csv')
results['risco_por_cluster'] = risco_cluster.reset_index().to_dict(orient='records')

perfil_norm = X_scaled_df.copy()
perfil_norm['cluster_kmeans'] = df['cluster_kmeans'].values
perfil_medias_norm = perfil_norm.groupby('cluster_kmeans')[features].mean()

fig, ax = plt.subplots(figsize=(10,6))
perfil_medias_norm.T.plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('Perfil Médio dos Clusters (Variáveis Padronizadas) – K-Means')
ax.set_ylabel('Valor médio padronizado (z-score)')
ax.set_xlabel('Variável')
ax.legend(title='Cluster')
ax.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig5_perfil_clusters.png', dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8,5))
taxa_inad_cluster = df.groupby('cluster_kmeans')['inadimplente'].mean()*100
bars = ax.bar(taxa_inad_cluster.index.astype(str), taxa_inad_cluster.values, color=sns.color_palette('viridis', K_OTIMO))
ax.axhline(results['taxa_inadimplencia_geral'], color='red', linestyle='--', label=f"Média geral ({results['taxa_inadimplencia_geral']:.1f}%)")
for bar, v in zip(bars, taxa_inad_cluster.values):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v:.1f}%', ha='center', fontsize=9)
ax.set_title('Taxa de Inadimplência por Cluster (K-Means)')
ax.set_xlabel('Cluster')
ax.set_ylabel('Taxa de Inadimplência (%)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIGS_OUT}/fig6_inadimplencia_cluster.png', dpi=150)
plt.close()

with open(f'{OUT}/resultados.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

df.to_csv(f'{OUT}/dataset_processado_com_clusters.csv', index=False)

print("\n\n=== RESUMO FINAL ===")
for k_, v_ in results.items():
    print(k_, ':', v_)
