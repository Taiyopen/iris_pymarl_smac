import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch
from torch.optim import RMSprop

import numpy as np

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        # 初始化 QLearner 類別
        self.args = args  # 儲存參數
        self.mac = mac  # 儲存多代理控制器
        self.logger = logger  # 儲存日誌記錄器

        self.params = list(mac.parameters())  # 獲取 MAC 的參數

        self.last_target_update_episode = 0  # 記錄上次更新目標網絡的回合

        self.mixer = None  # 初始化混合器
        if args.mixer is not None:
            # 根據參數選擇混合器
            if args.mixer == "vdn":
                self.mixer = VDNMixer()  # 使用 VDN 混合器
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)  # 使用 QMIX 混合器
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))  # 錯誤處理
            self.params += list(self.mixer.parameters())  # 添加混合器參數
            self.target_mixer = copy.deepcopy(self.mixer)  # 深拷貝混合器作為目標混合器

        # 初始化優化器
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # 深拷貝 MAC 作為目標 MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1  # 記錄日誌的時間戳

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        # 獲取相關數據
        rewards = batch["reward"][:, :-1]  # 獲取獎勵
        actions = batch["actions"][:, :-1]  # 獲取行動
        terminated = batch["terminated"][:, :-1].float()  # 獲取終止標誌
        mask = batch["filled"][:, :-1].float()  # 獲取填充掩碼
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])  # 更新掩碼
        avail_actions = batch["avail_actions"]  # 獲取可用行動

        # 計算估計的 Q 值
        mac_out = []  # 儲存 MAC 輸出
        self.mac.init_hidden(batch.batch_size)  # 初始化隱藏狀態
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)  # 獲取代理輸出
            mac_out.append(agent_outs)  # 添加到輸出列表
        mac_out = torch.stack(mac_out, dim=1)  # 在時間維度上堆疊

        # 獲取每個代理所採取行動的 Q 值
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # 移除最後一維

        x_mac_out = mac_out.clone().detach()  # 克隆 MAC 輸出
        x_mac_out[avail_actions == 0] = -9999999  # 將不可用行動的 Q 值設為極小值
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)  # 獲取最大 Q 值及其索引

        max_action_index = max_action_index.detach().unsqueeze(3)  # 添加維度
        is_max_action = (max_action_index == actions).int().float()  # 判斷是否為最大行動

        if show_demo:
            # 如果需要顯示示範數據
            q_i_data = chosen_action_qvals.detach().cpu().numpy()  # 獲取 Q 值數據
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()  # 計算 Q 值差異

        # 計算目標所需的 Q 值
        target_mac_out = []  # 儲存目標 MAC 輸出
        self.target_mac.init_hidden(batch.batch_size)  # 初始化目標 MAC 隱藏狀態
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)  # 獲取目標代理輸出
            target_mac_out.append(target_agent_outs)  # 添加到輸出列表

        # 不需要第一個時間步的 Q 值估計
        target_mac_out = torch.stack(target_mac_out[1:], dim=1)  # 在時間維度上堆疊

        # 在目標 Q 值上取最大值
        if self.args.double_q:
            # 使用雙 Q 學習獲取最大行動
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]  # 獲取當前最大行動
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)  # 獲取目標最大 Q 值
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]  # 獲取目標最大 Q 值

        # 混合 Q 值
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])  # 混合選擇的 Q 值
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])  # 混合目標最大 Q 值

        # 計算 1 步 Q 學習目標
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals  # 計算目標 Q 值

        if show_demo:
            # 如果需要顯示示範數據
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()  # 獲取總 Q 值數據
            tot_target = targets.detach().cpu().numpy()  # 獲取總目標數據
            if self.mixer == None:
                tot_q_data = np.mean(tot_q_data, axis=2)  # 計算平均 Q 值
                tot_target = np.mean(tot_target, axis=2)  # 計算平均目標

            # 輸出示範數據
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # 計算 TD 誤差
        td_error = (chosen_action_qvals - targets.detach())  # 計算 TD 誤差

        mask = mask.expand_as(td_error)  # 擴展掩碼以匹配 TD 誤差

        # 將來自填充數據的目標設為 0
        masked_td_error = td_error * mask  # 應用掩碼

        # 計算 L2 損失，對實際數據取平均
        loss = (masked_td_error ** 2).sum() / mask.sum()  # 計算損失

        masked_hit_prob = torch.mean(is_max_action, dim=2) * mask  # 計算命中概率
        hit_prob = masked_hit_prob.sum() / mask.sum()  # 計算平均命中概率

        # 優化
        self.optimiser.zero_grad()  # 清空梯度
        loss.backward()  # 反向傳播
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)  # 梯度裁剪
        self.optimiser.step()  # 更新參數

        # 更新目標網絡
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()  # 更新目標網絡
            self.last_target_update_episode = episode_num  # 更新上次更新回合

        # 記錄統計數據
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)  # 記錄損失
            self.logger.log_stat("hit_prob", hit_prob.item(), t_env)  # 記錄命中概率
            self.logger.log_stat("grad_norm", grad_norm, t_env)  # 記錄梯度範數
            mask_elems = mask.sum().item()  # 計算掩碼元素數量
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)  # 記錄 TD 誤差
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)  # 記錄採取的 Q 值平均
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)  # 記錄目標平均
            self.log_stats_t = t_env  # 更新日誌時間戳

    def _update_targets(self):
        # 更新目標網絡
        self.target_mac.load_state(self.mac)  # 加載 MAC 狀態到目標 MAC
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())  # 加載混合器狀態
        self.logger.console_logger.info("Updated target network")  # 記錄更新信息

    def cuda(self):
        # 將模型移動到 GPU
        self.mac.cuda()  # 移動 MAC 到 GPU
        self.target_mac.cuda()  # 移動目標 MAC 到 GPU
        if self.mixer is not None:
            self.mixer.cuda()  # 移動混合器到 GPU
            self.target_mixer.cuda()  # 移動目標混合器到 GPU

    def save_models(self, path):
        # 儲存模型
        self.mac.save_models(path)  # 儲存 MAC 模型
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.torch".format(path))  # 儲存混合器狀態
        torch.save(self.optimiser.state_dict(), "{}/opt.torch".format(path))  # 儲存優化器狀態

    def load_models(self, path):
        # 載入模型
        self.mac.load_models(path)  # 載入 MAC 模型
        # 不完全正確，但我不想儲存目標網絡
        self.target_mac.load_models(path)  # 載入目標 MAC 模型
        if self.mixer is not None:
            self.mixer.load_state_dict(torch.load("{}/mixer.torch".format(path), map_location=lambda storage, loc: storage))  # 載入混合器狀態
        self.optimiser.load_state_dict(torch.load("{}/opt.torch".format(path), map_location=lambda storage, loc: storage))  # 載入優化器狀態
