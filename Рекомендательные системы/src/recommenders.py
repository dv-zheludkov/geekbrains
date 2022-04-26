import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

from src.utils import prefilter_items



class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    def __init__(self, data, weighting=True, prefilter_n_popular=5000, prefilter_item_features=None):

        self.data = data
        n_items_before = self.data['item_id'].nunique()
        self.data = prefilter_items(self.data, take_n_popular=prefilter_n_popular, item_features=prefilter_item_features)
        n_items_after = self.data['item_id'].nunique()
        print('Decreased # items from {} to {}'.format(n_items_before, n_items_after))

        self.popularity = self.data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.popularity.sort_values('quantity', ascending=False, inplace=True)
        self.popularity = self.popularity[self.popularity['item_id'] != 999999]
        # self.popularity = self.popularity.groupby('user_id').head(5)
        # self.popularity.sort_values(by=['user_id', 'quantity'], ascending=False, inplace=True)
        # self.popularity = self.popularity.groupby('user_id')['item_id'].unique().reset_index()

        self.user_item_matrix = self.prepare_matrix(self.data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data: pd.DataFrame):

        # your_code
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity', # Можно пробоват ьдругие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )
        user_item_matrix = user_item_matrix.astype(float) # необходимый тип матрицы для implicit
        # print('user_item_matrix.head(3)', user_item_matrix.head(3))

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user_id, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        user_popularity = self.popularity[self.popularity.user_id == user_id].head(N).copy()
        user_popularity.sort_values(by=['user_id', 'quantity'], ascending=False, inplace=True)
        user_popularity = user_popularity.groupby('user_id')['item_id'].unique().reset_index()

        res = []
        try:
            for item_id in user_popularity.item_id.values[0]:
                # print('itemid_to_id:',  self.itemid_to_id[item_id])
                closest_items = [self.id_to_itemid[row_id] for row_id, score in self.model.similar_items(self.itemid_to_id[item_id], N=2)]
                res.append(closest_items[1])
        except:
            res.append('')

        # assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_users_recommendation(self, user_id, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        res = []
        try:
            similar_users = self.model.similar_users(self.userid_to_id[user_id], N=N)
            for similar_user in similar_users:
                top_1 = self.popularity[self.popularity.user_id == self.id_to_userid[similar_user[0]]]['item_id'].head(1).values[0]
                res.append(top_1)
        except:
            res.append('')

        # assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res