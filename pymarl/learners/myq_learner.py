import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch
from torch.optim import RMSprop

import numpy as np

class MyQLearner:
    def __init__(self, mac, scheme, logger, args):
        # 初始化 QLearner 類別
        self.args = args  # 儲存參數
        self.mac = mac  # 儲存多代理控制器
        self.logger = logger  # 儲存日誌記錄器

        self.params = list(mac.parameters())  # 獲取 MAC 的參數

        self.last_target_update_episode = 0  # 記錄上次更新目標網絡的回合

        self.mixer = None  # 初始化混合器

        # 初始化優化器
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.log_stats_t = -self.args.learner_log_interval - 1  # 記錄日誌的時間戳

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        # 獲取相關數據
        rewards = batch["reward"][:, :-1]  # 獲取獎勵
        actions = batch["actions"][:, :-1]  # 獲取行動
        dones = batch["terminated"][:, :-1].float()  # 獲取終止標誌
        mask = batch["filled"][:, :-1].float()  # 獲取填充掩碼
        mask[:, 1:] = mask[:, 1:] * (1 - dones[:, :-1])  # 更新掩碼

        
        mac_out = []  # 儲存多代理控制器的輸出
        self.mac.init_hidden(batch.batch_size)  # 初始化隱藏狀態
        for t in range(batch.max_seq_length):  # 遍歷序列長度
            agent_outs = self.mac.forward(batch, t=t)  # 獲取當前時間步的代理輸出
            mac_out.append(agent_outs)  # 將輸出添加到列表中
        mac_out = torch.stack(mac_out, dim=1)  # 將列表轉換為張量


        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # 獲取選擇行動的 Q 值


        targets = rewards + self.args.gamma * (1 - dones) * chosen_action_qvals  # 計算目標 Q 值

        loss = (targets - chosen_action_qvals).pow(2).sum() / mask.sum()  # 計算損失，使用均方誤差

        # 優化
        self.optimiser.zero_grad()  # 清空梯度
        loss.backward()  # 反向傳播計算梯度
        self.optimiser.step()  # 更新參數

        # 更新目標網絡
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:  # 檢查是否需要更新目標網絡
            self._update_targets()  # 更新目標網絡
            self.last_target_update_episode = episode_num  # 更新上次更新回合

    def _update_targets(self):
        pass

    def cuda(self):
        # 將模型移動到 GPU
        self.mac.cuda()  # 移動 MAC 到 GPU

    def save_models(self, path):
        # 儲存模型
        self.mac.save_models(path)  # 儲存 MAC 模型
        torch.save(self.optimiser.state_dict(), "{}/opt.torch".format(path))  # 儲存優化器狀態

    def load_models(self, path):
        # 載入模型
        self.mac.load_models(path)  # 載入 MAC 模型
        # 不完全正確，但我不想儲存目標網絡
        # self.target_mac.load_models(path)  # 去除載入目標 MAC 模型
        self.optimiser.load_state_dict(torch.load("{}/opt.torch".format(path), map_location=lambda storage, loc: storage))  # 載入優化器狀態
