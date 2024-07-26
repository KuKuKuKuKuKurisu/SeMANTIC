import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity

from .recommend_module import Encoder, DSI

from config import GlobalConfig, RecommendTrainConfig
if RecommendTrainConfig.simmc2:
    from config.simmc_dataset_config import SimmcDatasetConfig as DatasetConfig
else:
    from config.dataset_config import DatasetConfig
from config.constants import *
from utils.mask import *
from utils import load_pkl

class DST_AWARE(nn.Module):
    def __init__(self, pretrained_embedding, pretrained_image_encoder, state_keys):
        super(DST_AWARE, self).__init__()

        # constant
        self.text_emb_size = TEXT_EMB_SIZE
        self.layer_num = CROSS_LAYER_NUM
        self.head_num = CROSS_ATTN_HEAD_NUM

        self.img_emb_size = 512

        self.co_layer_num = CO_CROSS_LAYER_NUM

        self.co_emb_size = TEXT_EMB_SIZE + self.img_emb_size

        self.text_d_k = int(TEXT_EMB_SIZE/CROSS_ATTN_HEAD_NUM)

        self.padding_id = PADDING_ID

        # embedding layer
        self.word_embeddings = pretrained_embedding.word_embeddings
        self.position_embeddings = pretrained_embedding.position_embeddings

        # pretrained_image_encoder
        self.pretrained_image_encoder = pretrained_image_encoder
        if RecommendTrainConfig.simmc2:
            for param in self.pretrained_image_encoder.parameters():
                param.requires_grad = False

        # state_keys
        self.state_keys = state_keys.to(GlobalConfig.device)

        # text_encoder
        self.history_encoder = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                       d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        self.attrs_encoder = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                     d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        # image_encoder
        self.image_encoder = nn.Linear(self.img_emb_size, self.text_emb_size)

        # state aware emcoder
        self.state_encoder = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                     d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        self.get_state_aware = DSI(n_hop=3, d_model=self.text_emb_size)

        # pred state aware emcoder
        self.state_value_pred = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                        d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)
        self.state_learner = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                     d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        # context self_attention
        self.context_text_self_attention = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                                   d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        self.context_image_self_attention = Encoder(n_layers = self.layer_num, n_head = self.head_num, d_k = self.text_d_k, d_v = self.text_d_k,
                                                    d_model = self.text_emb_size, d_inner = self.text_emb_size, dropout = RecommendTrainConfig.dropout)

        # MSE
        self.mse = nn.MSELoss(reduction='mean')

        # kl
        self.kl = nn.KLDivLoss(reduction="none")

        # tanh
        self.tanh = nn.Tanh()

        # softmax
        self.softmax = nn.Softmax(dim = -1)

        # relu
        self.relu = nn.ReLU(inplace=False)

        # leaky relu
        self.leaky_relu = nn.LeakyReLU(inplace=False)

    def get_text_emb(self, text_ids, type):

        word_emb = self.word_embeddings(text_ids)
        pos_emb = self.position_embeddings(torch.tensor([pos for pos in range(len(text_ids[0]))]).repeat(len(text_ids), 1).to(GlobalConfig.device))

        word_emb = word_emb + pos_emb

        if type == 'history':
            attn_mask = get_attn_key_pad_mask(seq_k=text_ids, seq_q=text_ids, padding_id=self.padding_id)
            non_pad_mask = get_non_pad_mask(text_ids, self.padding_id)
            word_emb, = self.history_encoder(word_emb, word_emb, non_pad_mask, attn_mask)
        elif type == 'attrs':
            attn_mask = get_attn_key_pad_mask(seq_k=text_ids, seq_q=text_ids, padding_id=self.padding_id)
            non_pad_mask = get_non_pad_mask(text_ids, self.padding_id)
            word_emb, = self.attrs_encoder(word_emb, word_emb, non_pad_mask, attn_mask)

        return word_emb

    def get_image_emb(self, image):
        image_emb = self.pretrained_image_encoder(image).view(len(image), self.img_emb_size)
        image_emb = self.image_encoder(image_emb)
        return image_emb

    def get_state_emb(self, dst_ids, dst_attention_mask, batch_size, return_key_only = False):

        state_key_emb = self.word_embeddings(self.state_keys.unsqueeze(0)).repeat(batch_size, 1, 1)

        if return_key_only:
            return state_key_emb, None

        state_emb = self.word_embeddings(dst_ids)

        state_value_emb = []
        for key_id in range(len(state_key_emb[0])):
            unmask_num = torch.sum(dst_attention_mask[:, key_id * DatasetConfig.state_value_max_len : (key_id + 1) * DatasetConfig.state_value_max_len], dim = -1).view(-1, 1, 1)
            value_emb = state_emb[:, key_id * DatasetConfig.state_value_max_len : (key_id + 1) * DatasetConfig.state_value_max_len, :]
            value_emb = torch.sum(value_emb, dim = 1, keepdim=True)
            value_emb = value_emb/unmask_num
            state_value_emb.append(value_emb)

        state_value_emb = torch.cat(state_value_emb, dim = 1)
        state_emb = state_key_emb + state_value_emb

        attn_mask = None
        non_pad_mask = None
        state_emb, = self.state_encoder(state_emb, state_emb, non_pad_mask, attn_mask)

        return state_key_emb, state_emb

    def get_context_emb(self, text_ids, images, image_mask):

        # context text emb
        sent_emb = self.get_text_emb(text_ids, 'history')
        sent_emb = self.leaky_relu(torch.sum(sent_emb, dim = 1)).unsqueeze(1)

        # context image emb
        image_emb = []
        for img_id in range(len(images[0])):
            # image emb
            temp_image_emb = self.get_image_emb(images[:, img_id, :, :, :])
            temp_image_emb = temp_image_emb.unsqueeze(1)

            non_pad_mask = get_non_pad_mask(image_mask[:, img_id:img_id+1], self.padding_id)
            temp_image_emb *= non_pad_mask
            image_emb.append(temp_image_emb)

        image_emb = torch.cat(image_emb, dim = 1)
        image_emb = self.leaky_relu(torch.sum(image_emb, dim = 1)).unsqueeze(1)

        return sent_emb, image_emb

    def turn_level_self_attn(self, context_txt_emb, context_img_emb, turn_mask, state_key_embedding, state_embedding):

        attn_mask = get_attn_key_pad_mask(seq_k=turn_mask, seq_q=turn_mask, padding_id=self.padding_id)
        non_pad_mask = get_non_pad_mask(turn_mask, self.padding_id)
        context_txt_emb, = self.context_text_self_attention(context_txt_emb, context_txt_emb, non_pad_mask, attn_mask)
        context_img_emb, = self.context_image_self_attention(context_img_emb, context_txt_emb, non_pad_mask, attn_mask)

        # pred state embedding
        attn_mask = get_attn_key_pad_mask(seq_k=turn_mask, seq_q=state_key_embedding, padding_id=self.padding_id)
        non_pad_mask = None

        context_emb = context_txt_emb + context_img_emb
        pred_state_value_embedding, = self.state_value_pred(context_emb, state_key_embedding, non_pad_mask, attn_mask)
        pred_state_embedding = pred_state_value_embedding + state_key_embedding
        attn_mask = None
        non_pad_mask = None
        pred_state_embedding, = self.state_learner(pred_state_embedding, pred_state_embedding, non_pad_mask, attn_mask)

        # txt dst aware pred
        pred_context_txt_emb = self.get_state_aware(context_txt_emb[:, -1, :], pred_state_embedding, 'txt')
        # txt dst aware global truth
        if state_embedding != None:
            context_txt_emb = self.get_state_aware(context_txt_emb[:, -1, :], state_embedding, 'txt')

        # img dst aware pred
        pred_context_img_emb = self.get_state_aware(context_img_emb[:, -1, :], pred_state_embedding, 'img')
        # img dst aware global truth
        if state_embedding != None:
            context_img_emb = self.get_state_aware(context_img_emb[:, -1, :], state_embedding, 'img')

        if state_embedding != None:
            return pred_context_txt_emb, pred_context_img_emb, context_txt_emb, context_img_emb, pred_state_embedding
        else:
            return pred_context_txt_emb, pred_context_img_emb, None, None, pred_state_embedding

    def get_item_embedding(self, attrs_id, image, pred_state_emb, state_embedding):
        # text emb
        text_embedding = self.get_text_emb(attrs_id, 'attrs')
        text_embedding = self.leaky_relu(torch.sum(text_embedding, dim = 1))

        # image emb
        image_embedding = self.get_image_emb(image)
        image_embedding = self.leaky_relu(image_embedding)

        pred_text_embedding = self.get_state_aware(text_embedding, pred_state_emb, 'txt')
        if state_embedding != None:
            text_embedding = self.get_state_aware(text_embedding, state_embedding, 'txt')

        pred_image_embedding = self.get_state_aware(image_embedding, pred_state_emb, 'img')
        if state_embedding != None:
            image_embedding = self.get_state_aware(image_embedding, state_embedding, 'img')

        if state_embedding!= None:
            return pred_text_embedding, pred_image_embedding, text_embedding, image_embedding
        else:
            return pred_text_embedding, pred_image_embedding, None, None

    def get_kl_distribution(self, query_emb, key_emb):
        return self.softmax(cosine_similarity(query_emb.unsqueeze(1).repeat(1, len(key_emb[0]), 1), key_emb, dim = -1))

    def cal_kl_loss(self, input, target):
        return torch.sum(self.kl(input, target), dim = -1)

    def get_kl_loss(self, target_txt_kl, target_img_kl, input_txt_kl, input_img_kl):

        text_klloss = 0.5 * self.cal_kl_loss(torch.log(target_txt_kl), input_txt_kl) + \
                      0.5 * self.cal_kl_loss(torch.log(input_txt_kl), target_txt_kl)
        image_klloss = 0.5 * self.cal_kl_loss(torch.log(target_img_kl), input_img_kl) + \
                       0.5 * self.cal_kl_loss(torch.log(input_img_kl), target_img_kl)

        kl_loss = text_klloss + image_klloss
        return kl_loss

    def get_learn_loss(self, input, target):
        return self.mse(input, target)

    def forward(self, context_dialog, pos_products, neg_products, eval = False):

        texts, images, image_mask, utter_type, dst = context_dialog
        pos_image_num, pos_images, pos_product_texts = pos_products
        neg_image_num, neg_images, neg_product_texts = neg_products

        # to device
        text_ids = texts['input_ids'].to(GlobalConfig.device)
        text_type = texts['text_type'].to(GlobalConfig.device)
        turn_mask = texts['turn_mask'].to(GlobalConfig.device)

        pos_attrs_id = pos_product_texts['input_ids'].to(GlobalConfig.device)
        pos_attrs_type = pos_product_texts['text_type'].to(GlobalConfig.device)

        neg_attrs_id = neg_product_texts['input_ids'].to(GlobalConfig.device)
        neg_attrs_type = neg_product_texts['text_type'].to(GlobalConfig.device)

        images = images.to(GlobalConfig.device)

        image_mask = image_mask.to(GlobalConfig.device)

        pos_imgs_num = pos_image_num.to(GlobalConfig.device)
        neg_imgs_num = neg_image_num.to(GlobalConfig.device)

        pos_imgs = pos_images.to(GlobalConfig.device)
        neg_imgs = neg_images.to(GlobalConfig.device)

        # constants
        batch_size = len(images)
        ones = torch.ones(batch_size).to(GlobalConfig.device)
        zeros = torch.zeros(batch_size).to(GlobalConfig.device)

        # state embedding
        if RecommendTrainConfig.DST:
            dst_ids = dst['input_ids'].to(GlobalConfig.device)
            dst_attention_mask = dst['attention_mask'].to(GlobalConfig.device)

            state_key_embedding, state_embedding = self.get_state_emb(dst_ids, dst_attention_mask, batch_size)
        else:
            state_key_embedding, state_embedding = self.get_state_emb(None, None, batch_size, return_key_only = True)

        # context emb
        context_text_embeddings = []
        context_image_embeddings = []

        for turn_id in range(DatasetConfig.dialog_context_size):
            context_text_embedding, context_image_embedding = self.get_context_emb(text_ids[:, turn_id, :], images[:, turn_id, :, :, :, :], image_mask[:, turn_id, :])
            context_text_embeddings.append(context_text_embedding)
            context_image_embeddings.append(context_image_embedding)

        # dst or query aware
        pred_context_txt_emb, pred_context_img_emb, context_txt_emb, context_img_emb, \
        pred_state_emb = self.turn_level_self_attn(torch.cat(context_text_embeddings, dim = 1), torch.cat(context_image_embeddings, dim = 1), turn_mask,
                                                   state_key_embedding, state_embedding)
        # context kl
        if state_embedding!=None:
            # kl div
            pred_context_txt_emb, _ = pred_context_txt_emb
            pred_context_img_emb, _ = pred_context_img_emb
            context_txt_emb, context_text_kl = context_txt_emb
            context_img_emb, context_image_kl = context_img_emb
            # txt img sim
            context_text_image_sim = cosine_similarity(context_txt_emb, context_img_emb, dim = -1)
        else:
            # kl div
            pred_context_txt_emb, context_text_kl = pred_context_txt_emb
            pred_context_img_emb, context_image_kl = pred_context_img_emb
            # txt img sim
            context_text_image_sim = cosine_similarity(pred_context_txt_emb, pred_context_img_emb, dim = -1)

        context_text_image_sim = torch.max(zeros, RecommendTrainConfig.txt_img_threshold - context_text_image_sim)

        # sims
        neg_sims = []
        neg_text_image_sims = []
        neg_kl_losses = []

        pred_neg_sims = []

        # neg items
        for neg_id in range(DatasetConfig.max_neg_num):

            neg_pred_text_embedding, neg_pred_image_embedding, neg_text_embedding, neg_image_embedding = self.get_item_embedding(neg_attrs_id[:, neg_id, :],
                                                                                                                                 neg_imgs[:, neg_id, :, :, :],
                                                                                                                                 pred_state_emb, state_embedding)
            # neg sim
            pred_neg_text_sim = cosine_similarity(pred_context_txt_emb, neg_pred_text_embedding[0], dim = -1)
            pred_neg_image_sim = cosine_similarity(pred_context_img_emb, neg_pred_image_embedding[0], dim = -1)
            pred_neg_sims.append(self.tanh(pred_neg_text_sim + pred_neg_image_sim))
            if state_embedding!=None:
                neg_text_sim = cosine_similarity(context_txt_emb, neg_text_embedding[0], dim = -1)
                neg_image_sim = cosine_similarity(context_img_emb, neg_image_embedding[0], dim = -1)
                neg_sims.append(self.tanh(neg_text_sim + neg_image_sim))

            if eval == False:
                if state_embedding!=None:
                    # neg text image sim
                    neg_text_image_sim = cosine_similarity(neg_text_embedding[0], neg_image_embedding[0], dim = -1)
                    # neg kl
                    neg_text_kl = neg_text_embedding[1]
                    neg_image_kl = neg_image_embedding[1]
                else:
                    # neg text image sim
                    neg_text_image_sim = cosine_similarity(neg_pred_text_embedding[0], neg_pred_image_embedding[0], dim = -1)
                    # neg kl
                    neg_text_kl = neg_pred_text_embedding[1]
                    neg_image_kl = neg_pred_image_embedding[1]

                # neg txt img sim
                neg_text_image_sim = torch.max(zeros, RecommendTrainConfig.txt_img_threshold - neg_text_image_sim)
                neg_text_image_sims.append(neg_text_image_sim)

                # neg kl loss
                neg_kl_loss = self.get_kl_loss(context_text_kl, context_image_kl, neg_text_kl, neg_image_kl)
                neg_kl_losses.append(neg_kl_loss)

        # loss component
        mask = get_mask(DatasetConfig.max_neg_num, neg_imgs_num, GlobalConfig.device)
        losses = [[] for _ in range(DatasetConfig.max_pos_num)]
        pred_losses = [[] for _ in range(DatasetConfig.max_pos_num)]
        kl_losses = [[] for _ in range(DatasetConfig.max_pos_num)]

        pos_mask = get_mask(DatasetConfig.max_pos_num, pos_imgs_num, GlobalConfig.device)

        pos_text_image_sims = []

        # eval component
        rank_temp = torch.zeros(DatasetConfig.max_pos_num, batch_size, dtype=torch.long).to(GlobalConfig.device)
        pred_pos_sims = []

        for pos_id in range(DatasetConfig.max_pos_num):

            pos_pred_text_embedding, pos_pred_image_embedding, pos_text_embedding, pos_image_embedding = self.get_item_embedding(pos_attrs_id[:, pos_id, :],
                                                                                                                                 pos_imgs[:, pos_id, :, :, :],
                                                                                                                                 pred_state_emb, state_embedding)
            # pos sim
            pred_pos_text_sim = cosine_similarity(pred_context_txt_emb, pos_pred_text_embedding[0], dim = -1)
            pred_pos_image_sim = cosine_similarity(pred_context_img_emb, pos_pred_image_embedding[0], dim = -1)
            pred_pos_sim = self.tanh(pred_pos_text_sim + pred_pos_image_sim)

            if state_embedding!=None:
                pos_text_sim = cosine_similarity(context_txt_emb, pos_text_embedding[0], dim = -1)
                pos_image_sim = cosine_similarity(context_img_emb, pos_image_embedding[0], dim = -1)
                pos_sim = self.tanh(pos_text_sim + pos_image_sim)

            if eval == False:
                if state_embedding!=None:
                    # pos text image sim
                    pos_text_image_sim = cosine_similarity(pos_text_embedding[0], pos_image_embedding[0], dim = -1)
                    # pos kl
                    pos_text_kl = pos_text_embedding[1]
                    pos_image_kl = pos_image_embedding[1]
                else:
                    # pos text image sim
                    pos_text_image_sim = cosine_similarity(pos_pred_text_embedding[0], pos_pred_image_embedding[0], dim = -1)
                    # pos kl
                    pos_text_kl = pos_pred_text_embedding[1]
                    pos_image_kl = pos_pred_image_embedding[1]

                # neg txt img sim
                pos_text_image_sim = torch.max(zeros, RecommendTrainConfig.txt_img_threshold - pos_text_image_sim)
                pos_text_image_sims.append(pos_text_image_sim)

                # pos kl loss
                pos_kl_loss = self.get_kl_loss(context_text_kl, context_image_kl, pos_text_kl, pos_image_kl)

            pred_pos_sims.append(pred_pos_sim)
            for neg_id in range(len(pred_neg_sims)):
                # sim loss
                if eval:
                    loss = torch.max(zeros, ones - pred_pos_sim + pred_neg_sims[neg_id])
                    rank_temp[pos_id] += torch.lt(pred_pos_sim, pred_neg_sims[neg_id]).long() * mask[:, neg_id]
                else:
                    if state_embedding!= None:
                        loss = torch.max(zeros, ones - pos_sim + neg_sims[neg_id])
                        pred_loss = torch.max(zeros, ones - pred_pos_sim + pred_neg_sims[neg_id])
                        pred_losses[pos_id].append(pred_loss)
                    else:
                        loss = torch.max(zeros, ones - pred_pos_sim + pred_neg_sims[neg_id])

                    kl_loss = torch.max(zeros, pos_kl_loss - neg_kl_losses[neg_id])
                    kl_losses[pos_id].append(kl_loss)

                losses[pos_id].append(loss)

            # sim loss
            losses[pos_id] = torch.stack(losses[pos_id])
            losses[pos_id] = losses[pos_id].transpose(0, 1)
            losses[pos_id] = losses[pos_id].masked_select(mask.bool()).mean()

            if eval == False:
                # kl loss
                kl_losses[pos_id] = torch.stack(kl_losses[pos_id])
                kl_losses[pos_id] = kl_losses[pos_id].transpose(0, 1)
                kl_losses[pos_id] = kl_losses[pos_id].masked_select(mask.bool()).mean()

                # pred sim loss
                if state_embedding!= None:
                    pred_losses[pos_id] = torch.stack(pred_losses[pos_id])
                    pred_losses[pos_id] = pred_losses[pos_id].transpose(0, 1)
                    pred_losses[pos_id] = pred_losses[pos_id].masked_select(mask.bool()).mean()

        loss = 0
        count = 0
        for loss_tmp in losses:
            if loss_tmp != []:
                count += 1
                loss += loss_tmp
        loss /= count

        sim_loss = float(loss)

        if eval:
            # return loss, rank_temp, pos_imgs_num, pred_pos_sims, pred_neg_sims
            return loss, rank_temp, pos_imgs_num
        else:
            # add text image sim loss
            if RecommendTrainConfig.txt_img_loss:
                neg_text_image_sims = torch.stack(neg_text_image_sims, dim = 1)
                neg_text_image_sims = neg_text_image_sims.masked_select(mask.bool())
                pos_text_image_sims = torch.stack(pos_text_image_sims, dim = 1)
                pos_text_image_sims = pos_text_image_sims.masked_select(pos_mask.bool())
                image_text_sim_loss = torch.cat([neg_text_image_sims, pos_text_image_sims]).mean() # context_text_image_sim
                txt_img_loss = (image_text_sim_loss * RecommendTrainConfig.text_image_sim_loss_weight)
                loss += txt_img_loss
            else:
                txt_img_loss = 0

            # add kl loss
            if RecommendTrainConfig.kl_diff_loss:
                kl_loss = 0
                count = 0
                for loss_tmp in kl_losses:
                    if loss_tmp != []:
                        count += 1
                        kl_loss += loss_tmp
                kl_loss /= count
                loss += kl_loss * RecommendTrainConfig.diff_loss_weight
            else:
                kl_loss = 0

            # add learn loss
            if (state_embedding!=None) & (RecommendTrainConfig.start_learn):
                # pred sim loss
                pred_loss = 0
                count = 0
                for loss_tmp in pred_losses:
                    if loss_tmp != []:
                        count += 1
                        pred_loss += loss_tmp
                pred_loss /= count

                # state learn loss
                learn_loss = self.get_learn_loss(pred_state_emb, state_embedding)

                learn_loss += pred_loss
                loss = (1 - RecommendTrainConfig.learn_loss_weight) * loss + RecommendTrainConfig.learn_loss_weight * learn_loss
            else:
                learn_loss = 0

            return loss, (sim_loss, txt_img_loss, kl_loss, learn_loss, 0, 0)