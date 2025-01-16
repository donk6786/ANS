import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from models.BaseModel import GeneralModel

class ANS(GeneralModel):
    """
    ANS模型类，继承自GeneralModel。
    """
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['embedding_size']

    @staticmethod
    def parse_model_args(parser):
        """
        添加模型特定的参数解析。
        
        :param parser: ArgumentParser对象
        :return: 更新后的ArgumentParser对象
        """
        parser.add_argument('--embedding_size', type=int, default=64,
                          help='Size of embedding vectors.')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        """
        ANS模型初始化方法。
        
        :param args: 参数对象
        :param corpus: 语料库对象
        """
        super().__init__(args, corpus)
        self.embedding_size = args.embedding_size
        self.num_neg = args.num_neg
        self.user_num = corpus.n_users
        self.item_num = corpus.n_items

        self._define_params()
        self.apply(self.init_weights)

    def _define_params(self):
        """
        定义模型参数，包括用户和项目的嵌入矩阵。
        """
        # 定义嵌入层
        self.user_embedding = nn.Embedding(self.user_num, self.embedding_size)
        self.item_embedding = nn.Embedding(self.item_num, self.embedding_size)

    def forward(self, feed_dict):
        """
        前向传播函数，计算模型的输出。
        
        :param feed_dict: 包含输入数据的字典
        :return: 包含模型输出的字典
        """
        user_ids = feed_dict['user_id']  # [batch_size]
        item_ids = feed_dict['item_id']  # [batch_size, -1]

        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)

        # 使用 torch.einsum 计算预测值，更简洁地表示矩阵乘法和求和操作
        prediction = torch.einsum('bi,bji->bj', user_emb, item_emb)  # [batch_size, -1]
        user_emb = user_emb.repeat(1, item_ids.shape[1]).view(item_ids.shape[0], item_ids.shape[1], -1)
        item_emb = item_emb
        out_dict = {'prediction': prediction.view(feed_dict['batch_size'], -1),
                   'u_v': user_emb,
                   'i_v': item_emb}
        return out_dict

    def loss(self, out_dict):
        """
        计算模型的损失函数。
        
        :param out_dict: 包含模型输出的字典
        :return: 损失值
        """
        prediction = out_dict['prediction']
        pos_pred, neg_pred = prediction[:, 0], prediction[:, 1:]
        # 使用 BCEWithLogitsLoss 计算二分类交叉熵损失
        loss_fn = nn.BCEWithLogitsLoss()
        pos_labels = torch.ones_like(pos_pred)
        neg_labels = torch.zeros_like(neg_pred)
        labels = torch.cat([pos_labels[:, None], neg_labels], dim=1)
        loss = loss_fn(prediction, labels)
        return loss

    class Dataset(GeneralModel.Dataset):
        """
        ANS模型的数据集类，继承自GeneralModel的数据集类。
        """
        def __init__(self, model, corpus, phase):
            """
            数据集初始化方法。
            
            :param model: 模型对象
            :param corpus: 语料库对象
            :param phase: 阶段（'train', 'validate', 'test'）
            """
            super().__init__(model, corpus, phase)
            if self.phase == 'train':
                interaction_df = pd.DataFrame({
                    'user_id': self.data['user_id'],
                    'item_id': self.data['item_id']
                })
                all_item_ids = pd.Series(range(self.corpus.n_items), name='item_id')
                interaction_df = pd.concat([interaction_df, all_item_ids.to_frame()], ignore_index=True)
                self.interaction_df = interaction_df.drop_duplicates(subset=['item_id'])

        def _get_feed_dict(self, index):
            """
            根据索引获取feed_dict。
            
            :param index: 索引
            :return: feed_dict
            """
            user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
            if self.phase != 'train' and self.model.test_all:
                neg_items = np.arange(1, self.corpus.n_items)
            else:
                neg_items = self.data['neg_items'][index]
            item_ids = np.concatenate([[target_item], neg_items]).astype(int)
            feed_dict = {
                'user_id': user_id,
                'item_id': item_ids
            }
            return feed_dict

        def actions_before_epoch(self):
            """
            在每个epoch之前执行的动作，主要包括负采样。
            """
            neg_items = np.random.randint(1, self.corpus.n_items, size=(len(self), self.model.num_neg))
            for i, u in enumerate(self.data['user_id']):
                clicked_set = self.corpus.train_clicked_set[u]
                for j in range(self.model.num_neg):
                    while neg_items[i][j] in clicked_set:
                        # Augment negative sampling
                        neg_items[i][j] = self._augmented_negative_sample(u, clicked_set)
            self.data['neg_items'] = neg_items

        def _augmented_negative_sample(self, user_id, clicked_set):
            """
            增强负采样方法。
            
            :param user_id: 用户ID
            :param clicked_set: 用户点击的项目集合
            :return: 采样的项目ID
            """
            item_similarities = self._compute_item_popularity()
            available_items = list(set(range(self.corpus.n_items)) - clicked_set)
            weighted_items = [(item, item_similarities[item]) for item in available_items]
            # 使用 argsort 对相似度进行排序，提高性能
            sorted_indices = np.argsort([-weight for item, weight in weighted_items])
            sampled_item = available_items[sorted_indices[0]]
            return sampled_item

        def _compute_item_popularity(self):
            """
            计算项目的流行度。
            
            :return: 项目的流行度数组
            """
            item_interaction_count = self.interaction_df['item_id'].value_counts().reindex(range(self.corpus.n_items), fill_value=0).values

            if item_interaction_count.max() - item_interaction_count.min() == 0:
                popularity_norm = np.zeros_like(item_interaction_count)
            else:
                popularity_norm = (item_interaction_count - item_interaction_count.min()) / (item_interaction_count.max() - item_interaction_count.min())

            return popularity_norm