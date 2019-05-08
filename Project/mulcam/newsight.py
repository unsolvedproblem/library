#libraries
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

def readme(): 
    print("******Description*****")
    print("code by 현호킴, description by 승현백")
    print("클래스 이름.help() : 해당 클래스에서 사용할 수 있는 함수 출력")
    print("******Class names******")
    print("1) 데이터 불러오기 : Pickle2DF")
    print("2) 전처리 : PreprocessingText")
    print("3) 불용어,유의어 처리 : GetSimilarWords, GetStopWords")
    print("4) 문서 검색 :  GetDocsFromQuery")
    print("5) 벡터화 :  Vectorizer")
    print("6) 시각화 : Get2DPlot, AnalyzingNewsData")
    print("**********************")

# 1) 데이터 불러오기
class Pickle2DF:
    def help(self):
        print("******Pickle2DF******")
        print("1) get_dataframe('피클 경로') : 피클을 데이터프레임으로 반환")
        print("2) get_dataframe_from_list('피클 경로를 저장한 리스트') : 피클여러개를 데이터프레임으로 구성된 리스트로 반환")
        print("**********************")

    def get_dataframe(self, data_name_with_route):
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

# 2) 전처리
class PreprocessingText:
    def help(self):
        print("******PreprocessingText******")
        print("1) make_content_re('dataframe') : content 열 전처리 후, 'content_re'열에 저장")
        print("2) add_noun_dict('list') : 명사 사전에 단어 추가")
        print("3) add_stopwords('list') : 불용어 사전에 단어 추가")
        print("4) tokenize('dataframe') : 데이터 프레임에 'tokenized_doc' 열을 추가하고, 토큰화된 문서를 저장한다")
        print("*****************************")

    def __init__(self):
        self.reg_reporter = re.compile('[가-힣]+\s[가-힣]*기자')  # 기자
        self.reg_email = re.compile('[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$')  # 이메일
        self.reg_eng = re.compile('[a-z]+')  # 소문자 알파벳, 이메일 제거용, 대문자는 남겨둔다
        self.reg_chi = re.compile("[\u4e00-\u9fff]+")  # 한자
        self.reg_sc = re.compile("·|…|◆+|◇+|▶+|●+|▲+|“|”|‘|’|\"|\'|\(|\)")  # 특수문자
        self.reg_date = re.compile('\d+일|\d+월|\d+년|\d+시|\d+분|\(현지시간\)|\(현지시각\)|\d+')  # 날짜,시간,숫자
        
        self.twitter_obj = Twitter()
        self.stopwords = []
        self.noun_list = []

    def preprocessing(self, doc):
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
    
    def add_noun_dict(self,noun_list):
        self.twitter_obj.add_dictionary(noun_list, 'Noun')
        self.noun_list = self.noun_list+noun_list
        print("추가한 명사")
        print(noun_list)

    def add_stopwords(self,stopword_list):
        self.stopwords = self.stopwords+stopword_list
        print("추가한 불용어")
        print(stopword_list)

    def tokenize(self, data):
        print('추가한 명사:',self.noun_list)
        print('불용어: ', self.stopwords)
        tokenized_doc = data['content_re'].apply(lambda x: self.twitter_obj .nouns(x))
        tokenized_doc_without_stopwords = tokenized_doc.apply(
            lambda x: [item.lower() for item in x if item not in self.stopwords])
        data["tokenized_doc"] = tokenized_doc_without_stopwords
        return data

# 3) 불용어, 유의어 처리
class GetSimilarWords:
    def help(self):
        print("******GetSimilarWords******")
        print("1) get_model('토큰화된 문서 시리즈','doc2vec 후 차원 크기') : doc2vec 모델 학습")
        print("2) get_similar_words('단어') : 유의어 출력")
        print("*****************************")

    def get_model(self, tokenized_doc, size=300, window=5, min_count=5, workers=4, sg=1):
        self.model = Word2Vec(sentences=tokenized_doc, size=size, window=window, min_count=min_count, workers=workers,
                              sg=sg)
    def get_similar_words(self, string):
        print('단어 : 유사도')
        for word, score in self.model.wv.most_similar(string):
            print(f'{word} : {score}')

class GetStopWords:
    def help(self):
        print("******GetStopWords******")
        print("1)get_bow('토큰화된 문서 시리즈') : bow 생성")
        print("2)get_stop_words('단어 출현 빈도 순위  n ') : 단어 출현 빈도 상위 n 개, 하위 n 개 출력")
        print("*****************************")

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

# 4) 문서 검색
class GetDocsFromQuery:
    def help(self):
        print("******GetDocsFromQuery******")
        print("1)set_query('검색어') : 검색어 설정 ")
        print("2)select_news('토큰화된 문서 시리즈') : 검색어를 포함한 문서 시리즈 반환")
        print("*****************************")

    def set_query(self,query):
        self.query = query

    def select_news(self, tokenized_doc):
        selected_news = []
        for idx in range(len(tokenized_doc)):
            if self.query in tokenized_doc.iloc[idx]:
                selected_news.append(idx)
        print("length of selected news: ", len(selected_news))
        print("length of original data: ", len(tokenized_doc))
        return tokenized_doc.iloc[selected_news]

# 4) 벡터화
class Vectorizer:
    def help(self):
        print("******GetDocsFromQuery******")
        print("1)get_tfidf_vec('토큰화된 문서 시리즈','단어 수') : 문서를 tfidf 벡터(x) 와 단어(words)로 반환")
        print("2)get_doc2vec('토큰화된 문서 시리즈') : doc2vec 벡터 반환")
        print("3)load_doc2vec_model('토큰화된 문서 시리즈','모델객체이름') : 저장된 모델로  doc2vec 벡터 반환")    
        print("*****************************")

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

# 5) 시각화
class Get2DPlot:

    def help(self):
        print('아 난해하다 알아서 쓰세요')

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

    def help(self):
        print("******AnalyzingNewsData******")
        print("1)get_n_data_per_cluster('라벨 리스트') : 클러스터 별 문서 개수 반환")
        print("2)print_news_per_cluster('토큰화된 문서 시리즈','클러스터 라벨 리스트') : 클러스터 라벨별로 제목을 출력한다.")
        print("*****************************")


    def get_n_data_per_cluster(self, cluster_labels):
        tmp = pd.DataFrame(pd.Series(cluster_labels).value_counts(), columns=['counts'])
        tmp.index.name = 'cluster'
        return tmp

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