import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score, calinski_harabasz_score, \
    v_measure_score, fowlkes_mallows_score, silhouette_samples
import umap.umap_ as umap
from sklearn import metrics
from wordcloud import WordCloud
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import nltk
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
import re

nltk.download('omw-1.4')


class TrollTfidfVectorizer(TfidfVectorizer):

    def __init__(self, *args, **kwargs):
        troll_stop_words = {'don', 'just', 'like'}  # the custom stop word list could be further expanded
        kwargs['stop_words'] = set(ENGLISH_STOP_WORDS).union(troll_stop_words)
        kwargs['preprocessor'] = self.vectorizer_preprocess
        self.wnl = WordNetLemmatizer()
        super(TrollTfidfVectorizer, self).__init__(*args, **kwargs)

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: ([self.wnl.lemmatize(w) for w in analyzer(doc)])

    def vectorizer_preprocess(self, s):
        # remove urls
        s = re.sub(r'(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?', '', s)
        # remove amp
        s = s.replace('&amp;', '')
        # remove RT signs (no meaning) but keep username
        s = re.sub(r'\bRT\b\s+', '', s)
        s = s.lower()
        return s


def dim_reduction(embeddings):
    umap_embeddings = umap.UMAP(n_components=3,
                                n_neighbors=15,
                                random_state=42,
                                min_dist=0.01,
                                metric='cosine').fit_transform(embeddings)

    return umap_embeddings


def TFIDF_emb(df):
    vectorizer = TrollTfidfVectorizer(max_features=100, min_df=0.4)
    doc_term_matrix = vectorizer.fit_transform(df.Tweets.values)

    return doc_term_matrix, vectorizer


def clustering(df, embedding: list, n_clusters: int = 6):
    model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1)
    cluster_labels = model.fit_predict(embedding)
    dff = df.copy()
    dff["cluster_labels"] = cluster_labels

    return model, dff


def score_table(dfs, my_clusters, doc_term_matrix=None, embeddings=None):
    silhouette_avg = []
    purity_avg = []
    db_index = []
    vrc = []
    adjusted_rand = []
    v_measure = []
    fm_score = []

    if doc_term_matrix is None:
        for cluster in my_clusters:
            table = purity_silhouette(embeddings, dfs[cluster]["cluster_labels"], dfs[cluster]["Labels"])
            silhouette_avg.append(table.loc["silhouette"].mean())
            purity_avg.append(table.loc["purity"].mean())
            db_index.append(davies_bouldin_score(embeddings, dfs[cluster]["cluster_labels"]))
            vrc.append(calinski_harabasz_score(embeddings, dfs[cluster]["cluster_labels"]))
            adjusted_rand.append(adjusted_rand_score(dfs[cluster]["Labels"], dfs[cluster]["cluster_labels"]))
            v_measure.append(v_measure_score(dfs[cluster]["Labels"], dfs[cluster]["cluster_labels"]))
            fm_score.append(fowlkes_mallows_score(dfs[cluster]["Labels"], dfs[cluster]["cluster_labels"]))

        return (pd.DataFrame([silhouette_avg, purity_avg, db_index, vrc, adjusted_rand, v_measure, fm_score],
                             columns=["SBert_" + str(i) for i in my_clusters],
                             index=["silhouette_avg", "purity_avg", "db_index", "vrc", "adjusted_rand", "v_measure",
                                    "fm_score"]))

    else:
        doc_term_matrix = doc_term_matrix.toarray()
        for cluster in my_clusters:
            table = purity_silhouette(doc_term_matrix, dfs[cluster]["cluster_labels"], dfs[cluster]["Labels"])
            silhouette_avg.append(table.loc["silhouette"].mean())
            purity_avg.append(table.loc["purity"].mean())
            db_index.append(davies_bouldin_score(doc_term_matrix, dfs[cluster]["cluster_labels"]))
            vrc.append(calinski_harabasz_score(doc_term_matrix, dfs[cluster]["cluster_labels"]))
            adjusted_rand.append(adjusted_rand_score(dfs[cluster]["Labels"], dfs[cluster]["cluster_labels"]))
            v_measure.append(v_measure_score(dfs[cluster]["Labels"], dfs[cluster]["cluster_labels"]))
            fm_score.append(fowlkes_mallows_score(dfs[cluster]["Labels"], dfs[cluster]["cluster_labels"]))

    return (pd.DataFrame([silhouette_avg, purity_avg, db_index, vrc, adjusted_rand, v_measure, fm_score],
                         columns=["TF-IDF_" + str(i) for i in my_clusters],
                         index=["silhouette_avg", "purity_avg", "db_index", "vrc", "adjusted_rand", "v_measure",
                                "fm_score"]))


# Code taken from https://github.com/KIZI/evaluation-of-comprehensibility/blob/master/clustering_analysis_v2/LINVILLWARREN-AlternativeA-full.ipynb.ipynb
def purity_silhouette(data, cluster_labels, true_labels):
    sil = silhouette_samples(data, cluster_labels)
    silbycluster = []
    purbycluster = []
    for i in np.unique(cluster_labels):
        silbycluster.append(np.mean(sil[cluster_labels == i]))
        purbycluster.append(
            purity_score(np.array(true_labels)[cluster_labels == i], cluster_labels[cluster_labels == i]))
    silbycluster = np.round(np.array(silbycluster), 2)
    purbycluster = np.round(np.array(purbycluster), 2)
    return (pd.DataFrame([silbycluster, purbycluster], index=["silhouette", "purity"]))


# Code taken from https://github.com/KIZI/evaluation-of-comprehensibility/blob/master/clustering_analysis_v2/LINVILLWARREN-AlternativeA-full.ipynb.ipynb
def purity_score(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


# Code taken from https://github.com/KIZI/evaluation-of-comprehensibility/blob/master/clustering_analysis_v2/LINVILLWARREN-AlternativeA-full.ipynb.ipynb
def contingencytable(dataset_to_process, cluster_labels, true_labels):
    df = pd.DataFrame({'cluster': cluster_labels,
                       'category': true_labels})
    newdf = df.groupby(by=["cluster", "category"]).size().unstack().reset_index().drop("cluster", axis=1)
    newdf = newdf.rename_axis("cluster", axis="columns")
    fig = px.bar(newdf, x=newdf.index, y=newdf.columns.tolist(), title="Wide-Form Input")
    return fig


def run_fc(df, umap_emb=None, doc_term_matrix=None, n_clusters=6, vectorizer=None):
    if doc_term_matrix is None:

        model, dff = clustering(df, umap_emb, n_clusters)
        contingency_table = contingencytable(dff, dff.cluster_labels, dff.Labels)
        embeddings_plot = plot_embeddings(dff, umap_emb)
        wordcloud = generate_wordcloud(dff, horizontal=np.ceil(n_clusters / 2))
        wordcloud_url = fig_to_uri(wordcloud)

        return model, dff, contingency_table, embeddings_plot, wordcloud_url

    else:

        model, dff = clustering(df, doc_term_matrix, n_clusters)
        wordcloud = generate_wordcloud(dff, horizontal=np.ceil(n_clusters / 2), doc_term_matrix=doc_term_matrix,
                                       vectorizer=vectorizer)
        wordcloud_url = fig_to_uri(wordcloud)

        return dff, wordcloud_url


def plot_embeddings(df, umap_embeddings, x=0, y=1, z=2):
    fig = px.scatter_3d(df, x=umap_embeddings[:, x], y=umap_embeddings[:, y], z=umap_embeddings[:, z],
                        color=df["cluster_labels"].values)
    return fig


# Code modified from https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count


# Code modified from https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, column: str, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic[column])
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


# this function is used both for standard word clouds and z-score word clouds
def wordcloud_fn(dataframe, spectral=False, max_words=200, vertical=2, horizontal=3):
    if dataframe.shape[0] == 5:
        vertical = 1
        horizontal = 5

    clusters_word_freq = []

    for index, row in dataframe.iterrows():
        freq_dict = {}
        for col_name in dataframe.columns:
            if row[col_name] > 0.00001:
                freq_dict[col_name] = float(row[col_name])
        clusters_word_freq.append(freq_dict)

    fig = plt.figure(figsize=(20, 10))
    for cluster, freq_dict in enumerate(clusters_word_freq):
        if spectral:  # used for wordclouds from zscores, coolwarm goes from blue to red
            def color_func(word, *args, **kwargs):
                cmap = plt.cm.get_cmap('coolwarm')
                # Colormap instances are used to convert data values (floats) from the interval [0, 1] to the RGBA color
                rgb = cmap(freq_dict[word] / 100, bytes=True)[0:3]
                return rgb
        else:
            color_func = None

        ax = fig.add_subplot(int(vertical), int(horizontal), int(cluster + 1))
        cloud = WordCloud(normalize_plurals=False,
                          background_color='white', color_func=color_func, max_words=max_words, random_state=42)
        cloud.generate_from_frequencies(frequencies=freq_dict)
        ax.imshow(cloud, interpolation='bilinear')
        ax.set_yticks([])
        ax.set_xticks([])
        ax.text(0.35, 1, f'Cluster {cluster}',
                fontsize=32, va='bottom', transform=ax.transAxes)

    return fig


def dfFromModel(vectorizer, doc_term_matrix, cluster_labels):
    dataframe = DataFrame(doc_term_matrix.todense(), columns=vectorizer.get_feature_names())
    dataframe["labels"] = cluster_labels

    return dataframe.groupby('labels').mean()


def generate_wordcloud(df, MAX_WORDS_WORDCLOUD=100, MAX_FEATURES=None, MIN_DF=0.4, vertical=2, horizontal=4,
                       doc_term_matrix=None, vectorizer=None):
    if doc_term_matrix is None:
        vectorizer = TrollTfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
        doc_term_matrix = vectorizer.fit_transform(df.Tweets.values)
        fig = wordcloud_fn(dfFromModel(vectorizer, doc_term_matrix, df.cluster_labels), max_words=MAX_WORDS_WORDCLOUD,
                           vertical=vertical, horizontal=horizontal)
        return fig

    else:

        fig = wordcloud_fn(dfFromModel(vectorizer, doc_term_matrix, df.cluster_labels), max_words=MAX_WORDS_WORDCLOUD,
                           vertical=vertical, horizontal=horizontal)
        return fig


def fig_to_uri(in_fig, close_all=True, **save_args):
    """
    Save a figure as a URI
    :param in_fig:
    :return:
    """
    out_img = BytesIO()
    in_fig.savefig(out_img, format='png', **save_args)
    if close_all:
        in_fig.clf()
        plt.close('all')
    out_img.seek(0)
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/png;base64,{}".format(encoded)

def wordcloud_one_cluster(dataframe, my_cluster, max_words=200):
    if dataframe.shape[0] == 5:
        vertical = 1
        horizontal = 5

    clusters_word_freq = []

    for index, row in dataframe.iterrows():
        freq_dict = {}
        for col_name in dataframe.columns:
            if row[col_name] > 0.00001:
                freq_dict[col_name] = float(row[col_name])
        clusters_word_freq.append(freq_dict)

    fig = plt.figure(figsize=(20, 10))
    for cluster, freq_dict in enumerate(clusters_word_freq):
        if cluster == my_cluster:  # used for wordclouds from zscores, coolwarm goes from blue to red
            fig, ax = plt.subplots()
            cloud = WordCloud(normalize_plurals=False,
                              background_color='white', max_words=max_words, random_state=42)
            cloud.generate_from_frequencies(frequencies=freq_dict)
            ax.imshow(cloud, interpolation='bilinear')
            ax.set_yticks([])
            ax.set_xticks([])

            return fig


def generate_wordcloud_one_cluster(df, cluster, MAX_WORDS_WORDCLOUD=100, MAX_FEATURES=None, MIN_DF=0.4, vertical=2,
                                   horizontal=4, doc_term_matrix=None, vectorizer=None):
    if doc_term_matrix is None:
        vectorizer = TrollTfidfVectorizer(max_features=MAX_FEATURES, min_df=MIN_DF)
        doc_term_matrix = vectorizer.fit_transform(df.Tweets.values)
        fig = wordcloud_one_cluster(dfFromModel(vectorizer, doc_term_matrix, df.cluster_labels), cluster,
                                    max_words=MAX_WORDS_WORDCLOUD)
        return fig
    else:
        fig = wordcloud_one_cluster(dfFromModel(vectorizer, doc_term_matrix, df.cluster_labels), cluster,
                                    max_words=MAX_WORDS_WORDCLOUD)
        return fig


def fact_checking_assignment(embedding_df, embedding_fact_checker, dataset, n_clusters=5):

    embedding = np.concatenate([embedding_df, embedding_fact_checker], axis=0)
    umap_emb = dim_reduction(embedding)
    model = KMeans(n_clusters= n_clusters, init='k-means++', n_init=1)
    cluster_labels = model.fit_predict(umap_emb)
    my_cluster = cluster_labels[-1]
    dff = df.copy()
    dff["cluster_labels"] = cluster_labels[:-1]
    fig = generate_wordcloud_one_cluster(x, my_cluster, MAX_WORDS_WORDCLOUD =100, MAX_FEATURES = None, MIN_DF = 0.4, doc_term_matrix = None, vectorizer = None)

    return int(my_cluster), dff[dff.cluster_labels == my_cluster], fig