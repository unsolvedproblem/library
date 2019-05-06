"""
뉴스 자동분류 서비스
1. 토큰화
2. word2vec, bow
3. 새로운 query를 포함한 문서 추출
4. tf-idf, tf-idf 클러스터링, kmeans, dbscan, 시각화
5. doc2vec, doc2vec 클러스터링, kmeans, dbscan, 시각화
6. tf-idf+ count
- 멀캠 뉴사이트
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pickle
import datetime
import re
from ckonlpy.tag import Twitter
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

matplotlib.rc('font', family='NanumBarunGothic')
plt.rcParams['axes.unicode_minus'] = False


class MulCamNewSightTokenize:
    def __init__(self, add_dict=None, stopwords=None):
        self.reg_reporter = re.compile('[가-힣]+\s[가-힣]*기자')  # 기자
        self.reg_email = re.compile('[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')  # 이메일
        self.reg_eng = re.compile('[a-z]+')  # 소문자 알파벳, 이메일 제거용, 대문자는 남겨둔다
        self.reg_chi = re.compile("[\u4e00-\u9fff]+")  # 한자
        self.reg_sc = re.compile("·|…|◆+|◇+|▶+|●+|▲+|“|”|‘|’|\"|\'|\(|\)")  # 특수문자
        self.reg_date = re.compile('\d+일|\d+월|\d+년|\d+시|\d+분|\(현지시간\)|\(현지시각\)|\d+')  # 날짜,시간,숫자
        self.add_dict = add_dict
        self.stopwords = stopwords

    def get_dataframe(self, data_name_with_route):
        """
        load news data
        """
        with open(data_name_with_route, 'rb') as file:
            data_list = []
            while True:
                try:
                    data = pickle.load(file)
                except EOFError:
                    break
                data_list.append(data)
        # construct lists for data frame
        title = []
        content = []
        date = []
        for news in data_list[0]['return_object']['documents']:
            title.append(news['title'])
            content.append(news['content'])
            date.append(news['published_at'][:10])  # 시간 조정이 필요하면 바꾸기
        # make lists as data frame
        news_data = pd.DataFrame([])
        news_data['date'] = date
        news_data['title'] = title
        news_data['content'] = content
        news_data['date_tmp'] = news_data['date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').toordinal())
        return news_data

    def get_dataframe_from_list(self, data_names):
        """
        주소가 저장되어있는 리스트에서 data frame 만들기
        """
        news_data_list = []
        for data_name in data_names:
            news_data = self.get_dataframe(data_name)
            news_data_list.append(news_data)
        data = pd.DataFrame([])
        for news_data in news_data_list:
            data = data.append(news_data)
        data.reset_index(inplace=True)
        data.drop(['index'], axis=1, inplace=True)
        return data

    def preprocessing(self, doc):
        """
        정규표현식 전처리 함수
        """
        tmp = re.sub(self.reg_reporter, '', doc)
        tmp = re.sub(self.reg_email, '', tmp)
        tmp = re.sub(self.reg_eng, '', tmp)
        tmp = re.sub(self.reg_chi, '', tmp)
        tmp = re.sub(self.reg_sc, ' ', tmp)
        tmp = re.sub(self.reg_date, '', tmp)
        return tmp

    def make_content_re(self, data):
        data['content_re'] = data['content'].apply(self.preprocessing)
        return data

    def show_add_dict(self):
        print('add_dict: ', self.add_dict)

    def show_stopwords(self):
        print('stopwords: ', self.stopwords)

    def tokenize(self, data):
        print('add_dict:', self.add_dict)
        print('불용어: ', self.stopwords)
        twitter = Twitter()
        twitter.add_dictionary(self.add_dict, 'Noun')
        tokenized_doc = data['content_re'].apply(lambda x: twitter.nouns(x))
        tokenized_doc_without_stopwords = tokenized_doc.apply(
            lambda x: [item.lower() for item in x if item not in self.stopwords])
        data["tokenized_doc"] = tokenized_doc_without_stopwords
        return data


class GetSimilarWords(Word2Vec):
    def get_model(self, tokenized_doc, size=300, window=5, min_count=5, workers=4, sg=1):
        self.model = Word2Vec(sentences=tokenized_doc, size=size, window=window, min_count=min_count, workers=workers,
                              sg=sg)

    def get_similar_words(self, string):
        for word, score in self.model.wv.most_similar(string):
            print(word)


class GetStopWords:
    def get_bow(self, sentences):
        self.tmp_list = []
        for doc in sentences:
            self.tmp_list.extend(doc)
        self.word_count = pd.Series(self.tmp_list).value_counts()
        self.word_count_idx = list(self.word_count.index)

    def get_stop_words(self, number):
        stop_words_candi = self.word_count_idx[:number] + self.word_count_idx[-number:]
        for word in stop_words_candi:
            print(word)


class GetDocsFromQuery:
    def __init__(self, query):
        self.query = query

    def select_news(self, tokenized_doc):
        selected_news = []
        for idx in range(len(tokenized_doc)):
            if self.query in tokenized_doc.iloc[idx]:
                selected_news.append(idx)
        print("length of selected news: ", len(selected_news))
        print("length of original data: ", len(tokenized_doc))
        return tokenized_doc.iloc[selected_news]


class Vectorizer:
    def get_tfidf_vec(self, query_doc, max_feat=None):
        query_doc = query_doc.apply(lambda x: ' '.join(x))
        obj = TfidfVectorizer(max_features=max_feat)  # max_features for lda
        x = obj.fit_transform(query_doc).toarray()
        words = np.array(obj.get_feature_names())
        return x, words

    def get_doc2vec(self, query_doc,
                    dm=1, dbow_words=1, window=8, vector_size=50, alpha=0.025,
                    seed=42, min_count=5, min_alpha=0.025, workers=4, hs=0, negative=10,
                    n_epochs=50, model_name='d2v.model'):
        tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(query_doc)]
        model = Doc2Vec(
            dm=dm,  # PV-DBOW => 0 / default 1
            dbow_words=dbow_words,  # w2v simultaneous with DBOW d2v / default 0
            window=window,  # distance between the predicted word and context words
            vector_size=vector_size,  # vector size
            alpha=alpha,  # learning-rate
            seed=seed,
            min_count=min_count,  # ignore with freq lower
            min_alpha=min_alpha,  # min learning-rate
            workers=workers,  # multi cpu
            hs=hs,  # hierarchical softmax / default 0
            negative=negative,  # negative sampling / default 5
        )
        model.build_vocab(tagged_data)
        print("corpus_count: ", model.corpus_count)

        for epoch in range(n_epochs):
            if epoch % 10 == 0:
                print('epoch: ', epoch)
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            model.alpha -= 0.0002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        model.save(model_name)
        print("Model Saved")
        model_loaded = Doc2Vec.load(model_name)
        print("Load Model")
        x_doc2vec = []
        for i in range(len(query_doc)):
            x_doc2vec.append(model_loaded.docvecs[i])
        x_doc2vec = np.array(x_doc2vec)
        return x_doc2vec

    def load_doc2vec_model(self, query_doc, model_name):
        print("Load Model")
        model_loaded = Doc2Vec.load(model_name)
        x_doc2vec = []
        for i in range(len(query_doc)):
            x_doc2vec.append(model_loaded.docvecs[i])
        x_doc2vec = np.array(x_doc2vec)
        return x_doc2vec


class Get2DPlot:
    def __init__(self, x, reduction_method="PCA",
                 learning_rate=200, init='pca', random_state=10,
                 cluster_method="DBSCAN", eps=7, min_sample=2, n_clusters=3):
        self.reduction_method = reduction_method
        self.learning_rate = learning_rate
        self.init = init
        self.random_state = random_state
        self.cluster_method = cluster_method
        self.eps = eps
        self.min_sample = min_sample
        self.n_clusters = n_clusters
        self.x_scaled = StandardScaler().fit_transform(x)

    def get_2D_vec(self):
        if self.reduction_method == 'TSNE':
            t_sne = TSNE(n_components=2, learning_rate=self.learning_rate, init=self.init,
                         random_state=self.random_state)
            self.vec = t_sne.fit_transform(self.x_scaled)
        elif self.reduction_method == 'PCA':
            pca = PCA(n_components=2)
            self.vec = pca.fit_transform(self.x_scaled)
        return self.vec

    def get_cluster_labels(self):
        if self.cluster_method == 'kmeans':
            cluster = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.x_scaled)
            self.cluster_labels = cluster.labels_
        elif self.cluster_method == 'DBSCAN':
            self.cluster_labels = DBSCAN(eps=self.eps, min_samples=self.min_sample).fit_predict(self.x_scaled)
        vec_pd = np.c_[self.vec, self.cluster_labels]
        self.vec_pd = pd.DataFrame(vec_pd, columns=['x', 'y', 'labels'])
        print(self.cluster_labels)

    def plot2D(self):
        print(self.reduction_method, self.cluster_method)
        groups = self.vec_pd.groupby('labels')
        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(group.x,
                    group.y,
                    marker='o',
                    linestyle='',
                    label=name)
        # ax.legend(fontsize=12, loc='upper left') # legend position

        plt.title('%s Plot of %s' % (self.reduction_method, self.cluster_method), fontsize=20)
        plt.xlabel('x', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.show()


class AnalyzingNewsData:
    def print_news_per_cluster(self, data, clusters, content='title'):
        for n in clusters:
            print('=' * 100, n)
            for i in range(len(data)):
                try:
                    print(data[data['labels'] == n][content].iloc[i])
                    print('*' * 100)
                except:
                    break
            print('=' * 100, n)

