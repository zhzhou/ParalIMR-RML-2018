import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import IncrementalNet
from utils.inc_net import CosineIncrementalNet
from utils.toolkit import target2onehot, tensor2numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.font_manager as fm

# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# 1. 指定你的字体文件相对路径（假设你把它传到了项目根目录）
font_path = './simhei.ttf'

# 2. 将字体动态添加到 Matplotlib 的字体管理器中
fm.fontManager.addfont(font_path)

# 3. 获取该字体在底层的真实名称（防止大小写或内部命名差异）
prop = fm.FontProperties(fname=font_path)
font_name = prop.get_name()

# 4. 全局应用该字体，并解决负号显示问题
plt.rcParams['font.sans-serif'] = [font_name]
plt.rcParams['axes.unicode_minus'] = False

# 创新一完整代码，与创新点二无关系

EPSILON = 1e-8

init_epoch = 80

init_lr = 0.1
init_milestones = [100]
init_lr_decay = 0.1
init_weight_decay = 0.0005

epochs = 80
lrate = 0.001
milestones = [50]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4
T = 2


class iCaRL(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, False)

        self.snrs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        self.acc_pred_snr = torch.zeros(len(self.snrs))
        self.acc_total_snr = torch.zeros(len(self.snrs))
        self.acc_snr = torch.zeros(len(self.snrs))
        self.ref_signals = None

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning4 on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.random_matrix = torch.tensor(data_manager.Random_Matrix).float()
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.clean_data = torch.tensor(
            np.array(data_manager.transformed_clean_data),  # 先转为 numpy 数组
            device=self._device
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        print("Analyzing Prototype Space Topology...")
        current_dists = self.analyze_inter_class_separation(self._cur_task)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        logging.info(f"Generating confusion matrix for Task {self._cur_task}...")
        self.generate_and_plot_confusion_matrix(self.test_loader)


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        self.acc_pred_snr = torch.zeros(len(self.snrs))
        self.acc_total_snr = torch.zeros(len(self.snrs))
        self.acc_snr = torch.zeros(len(self.snrs))

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )  # 1e-5
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )

            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(init_epoch))
        criterion_reconstruction = nn.MSELoss()  # 重构损失（自编码器）

        A, B = [], []
        self.last_5_epochs_acc = []
        self.snr_to_idx = {snr: idx for idx, snr in enumerate(self.snrs)}
        alpha = 0

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            recon_loss = 0.0  # 用于记录自编码器损失
            correct, total = 0, 0

            current_acc_pred_snr = [0] * len(self.snrs)
            current_acc_total_snr = [0] * len(self.snrs)

            if epoch < 50:
                alpha = 0.0
            elif epoch < 55:
                alpha = 0.01  # 初期用小权重引入
            else:
                alpha = 0.1  # 后期加大权重

            # for i, (_, inputs, targets, snr) in enumerate(train_loader):
            for i, (idx, inputs, targets, snr) in enumerate(train_loader):
                inputs = self.discrete_random_mixing(inputs, targets, snr)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                clean_inputs = self.clean_data[idx].to(self._device)

                outputs = self._network(inputs)
                logits = outputs["logits"]
                denoised = outputs["denoised"]  # 降噪自编码器的输出

                # with torch.no_grad():
                #     f_clean = self._network(clean_inputs)["features"]

                # 计算分类损失
                loss = F.cross_entropy(logits, targets.long())
                recon_loss = F.mse_loss(denoised, clean_inputs)
                # recon_loss = F.mse_loss(denoised, f_clean)
                # 总损失是分类损失 + 重构损失的加权和]
                # if epoch <= 50:
                #     total_loss = loss
                # else:
                #     total_loss = loss + 0.1 * recon_loss  # 这里的0.1是加权系数，调整它以控制两个损失的平衡

                total_loss = loss + alpha * recon_loss  # 这里的0.1是加权系数，调整它以控制两个损失的平衡

                # total_loss = loss
                # 反向传播和优化
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                losses += loss.item()  # 分类损失
                # ae_losses += recon_loss.item()  # 自编码器损失

                # 计算准确率
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                # 遍历样本时更新当前epoch的统计（使用当前epoch的临时变量）
                for idx_sample in range(targets.shape[0]):
                    snr_val = snr[idx_sample].item()
                    if snr_val not in self.snr_to_idx:
                        continue
                    idx = self.snr_to_idx[snr_val]
                    current_acc_pred_snr[idx] += (targets[idx_sample] == preds[idx_sample]).cpu().item()
                    current_acc_total_snr[idx] += 1

            # 计算当前epoch每个SNR的准确率
            current_epoch_acc = []
            for j in range(len(self.snrs)):
                acc = current_acc_pred_snr[j] / current_acc_total_snr[j] if current_acc_total_snr[j] > 0 else 0.0
                current_epoch_acc.append(acc)

            # 保存当前epoch结果到列表，只保留最后5个
            self.last_5_epochs_acc.append(current_epoch_acc)
            if len(self.last_5_epochs_acc) > 5:
                self.last_5_epochs_acc.pop(0)  # 移除最早的epoch结果

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            A.extend([train_acc])

            if epoch % 2 == 0:
                # test_acc = self._compute_accuracy(self._network, test_loader)
                test_acc, test_acc_snr, group_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, AE_Loss {:.8f}, Train_accy {:.3f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    loss,
                    recon_loss,  # 输出自编码器的平均损失
                    train_acc,
                    test_acc,
                )

                B.append([test_acc])
                # //
                print("Test_accy = ", test_acc)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, AE_Loss {:.8f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    loss,
                    recon_loss,  # 输出自编码器的平均损失
                    train_acc,
                )

            prog_bar.set_description(info)

        # 训练全部结束后，计算最后5个epoch的平均值
        self.avg_last_5_acc = []
        for j in range(len(self.snrs)):
            # 对每个SNR，取最后5个epoch的准确率平均值
            sum_acc = sum(epoch_acc[j] for epoch_acc in self.last_5_epochs_acc)
            avg_acc = sum_acc / len(self.last_5_epochs_acc)
            self.avg_last_5_acc.append(avg_acc)

        print(A)
        print(B)
        print(self.avg_last_5_acc)
        print(test_acc_snr)
        print(group_acc)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(epochs))
        A, B = [], []
        self.last_5_epochs_acc = []
        self.snr_to_idx = {snr: idx for idx, snr in enumerate(self.snrs)}
        old_num = 0

        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            recon_loss = 0

            current_acc_pred_snr = [0] * len(self.snrs)
            current_acc_total_snr = [0] * len(self.snrs)
            if epoch < 10:
                alpha = 0.0
            elif epoch < 30:
                alpha = 0.02  # 初期用小权重引入
            else:
                alpha = 0.5  # 后期加大权重

            for i, (idx, inputs, targets, snr) in enumerate(train_loader):
                inputs = self.discrete_random_mixing(inputs, targets, snr)
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                clean_inputs = self.clean_data[idx].to(self._device)
                outputs = self._network(inputs)
                logits = outputs["logits"]
                denoised = outputs["denoised"]  # 降噪自编码器的输出

                # 1. 获取当前张量所在的设备 (CPU 或 GPU)，确保权重张量和数据在同一个设备上
                device = logits.device

                # 2. 初始化一个长度为 8 的全 1 张量 (对应 0-7 类)
                if self._cur_task == 1:
                    old_num = 8
                else:
                    old_num = 11

                class_weights = torch.ones(old_num, device=device)

                # 3. 施加非对称惩罚：赋予旧类 (0-4) 更高的权重
                # 这里的 1.5 是一个超参数，你可以根据实验效果在 1.2 到 2.0 之间微调
                class_weights[0:old_num - 3] = 1.75
                class_weights[old_num - 3:old_num] = 1.0  # 新类保持正常的 1.0 权重

                # --- 替换你原来的 loss 计算 ---

                # 4. 把权重传入 cross_entropy 函数中
                loss_clf = F.cross_entropy(logits, targets.long(), weight=class_weights)
                # # 1. 分类损失：使用扣分后的 logits_am
                # loss_clf = F.cross_entropy(logits, targets.long())

                loss_kd = (T * T) * _KD_loss(
                    logits[:, : self._known_classes],
                    self._old_network(inputs)["logits"],
                    T,
                )

                recon_loss = F.mse_loss(denoised, clean_inputs)
                loss = loss_clf + 0.5 * loss_kd

                total_loss = loss_clf + loss_kd + alpha * recon_loss

                # loss_clf = F.cross_entropy(logits, targets.long())
                # loss_kd = (T * T) * _KD_loss(
                #     logits[:, : self._known_classes],
                #     self._old_network(inputs)["logits"],
                #     T,
                # )
                # recon_loss = F.mse_loss(denoised, clean_inputs)
                # loss = loss_clf + loss_kd
                # total_loss = loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

                for idx_sample in range(targets.shape[0]):
                    snr_val = snr[idx_sample].item()
                    if snr_val not in self.snr_to_idx:
                        continue
                    idx = self.snr_to_idx[snr_val]
                    current_acc_pred_snr[idx] += (targets[idx_sample] == preds[idx_sample]).cpu().item()
                    current_acc_total_snr[idx] += 1

            # 计算当前epoch每个SNR的准确率
            current_epoch_acc = []
            for j in range(len(self.snrs)):
                acc = current_acc_pred_snr[j] / current_acc_total_snr[j] if current_acc_total_snr[j] > 0 else 0.0
                current_epoch_acc.append(acc)

            # 保存当前epoch结果到列表，只保留最后5个
            self.last_5_epochs_acc.append(current_epoch_acc)
            if len(self.last_5_epochs_acc) > 5:
                self.last_5_epochs_acc.pop(0)  # 移除最早的epoch结果

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            A.extend([train_acc])

            if epoch % 2 == 0:
                # test_acc = self._compute_accuracy(self._network, test_loader)
                test_acc, test_acc_snr, group_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Ep {}/{} => L_Clf {:.3f}, Tr_Acc {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, epochs,
                    loss,
                    train_acc,
                    test_acc
                )

                B.append([test_acc])
                # //
                print("Test_accy = ", test_acc)
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, AE_Loss {:.8f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    loss,
                    recon_loss,  # 输出自编码器的平均损失
                    train_acc,
                )

            prog_bar.set_description(info)

        # 训练全部结束后，计算最后5个epoch的平均值
        self.avg_last_5_acc = []
        for j in range(len(self.snrs)):
            # 对每个SNR，取最后5个epoch的准确率平均值
            sum_acc = sum(epoch_acc[j] for epoch_acc in self.last_5_epochs_acc)
            avg_acc = sum_acc / len(self.last_5_epochs_acc)
            self.avg_last_5_acc.append(avg_acc)

        print(A)
        print(B)
        print(self.avg_last_5_acc)
        print(test_acc_snr)
        print(group_acc)

        logging.info(info)


    # p=0.0625
    # def discrete_random_mixing(self, X_train, Y_train, Z_train, p=0.0625, low_snr=True):
    #     # 计算需要替换的离散位置数量
    #     num = int(X_train.shape[2] * p)
    #     # 克隆原始数据
    #     res = X_train.clone()
    #     # 获取信噪比级别总数
    #     # snr_len = len(self.snrs)
    #     snr_len = 20
    #
    #     # row = self._total_classes * snr_len
    #     # 生成行索引矩阵（每个样本的索引重复num次）
    #     # 形状：(batch_size, num)
    #     row_index = np.array([[i for i in range(X_train.shape[0])] for _ in range(num)]).transpose()
    #     # [[0....0][1...1][2....2].....[128....128]]
    #
    #     # 为每个样本随机选择num个离散的列位置（允许重复）(batch_size, num)
    #     res_index_col = np.random.choice(range(X_train.shape[2]), size=[X_train.shape[0], num], replace=True)
    #
    #     # 从随机矩阵中随机选择列位置（允许重复）Random_Matrix:[mods*snrs , 2 , 128*50]
    #
    #     random_matrix_index_col = np.random.choice(range(self.random_matrix.shape[2]), size=[X_train.shape[0], num],
    #                                                replace=True)
    #     if low_snr == False:
    #         # 同一调制类型和相同信噪比的随机样本
    #         random_matrix_index_row = (Y_train * snr_len + (Z_train + 20) / 2).reshape(
    #             [Y_train.shape[0], 1]).cpu().numpy()
    #     else:
    #         # 低信噪比模式：添加随机偏移
    #         # 生成形状：(batch_size, 1)的基础行号
    #         # 来源同一调制类型，不同的低信噪比样本
    #         random_matrix_index_row = (
    #                 (Y_train * snr_len).cpu().numpy().reshape([X_train.shape[0], 1]) + np.random.randint(
    #             np.zeros(X_train.shape[0]), ((Z_train + 20) / 2).cpu().numpy() + 1, [X_train.shape[0], 1]))
    #     # 扩展行索引矩阵到与列数匹配的形状（batch_size, num）
    #     ones = np.ones([1, num])
    #     random_matrix_index_row = (random_matrix_index_row * ones).astype(int)
    #     res[row_index, :, res_index_col] = self.random_matrix[random_matrix_index_row, :, random_matrix_index_col]
    #     # 执行替换操作（三维索引）
    #     # row_index: (batch_size, num) 每个元素是样本索引
    #     # res_index_col: (batch_size, num) 每个样本要替换的列位置
    #     # random_matrix_index_row: (batch_size, num) 随机矩阵的行索引
    #     # random_matrix_index_col: (batch_size, num) 随机矩阵的列索引
    #     return res

    def discrete_random_mixing(self, X_train, Y_train, Z_train, p=0.0625, low_snr=True):
        """
        基于离散点替换的混合增强策略 (带自适应功率缩放)
        """
        # 1. 基础参数准备
        batch_size = X_train.shape[0]
        seq_len = X_train.shape[2]  # 通常是 128
        num = int(seq_len * p)  # 替换点的数量，例如 128*0.0625 = 8
        snr_len = 20  # 信噪比阶数 (-20dB 到 18dB 共20个)

        # 2. 克隆数据，避免修改原始输入引用
        res = X_train.clone()

        # 确保 random_matrix 在正确的设备上 (GPU/CPU)
        if self.random_matrix.device != X_train.device:
            self.random_matrix = self.random_matrix.to(X_train.device)

        # 3. 生成目标位置索引 (Target Indices)
        # 为每个样本随机选择 num 个要被替换的时间点列索引
        # res_index_col: [batch_size, num]
        res_index_col = np.random.choice(range(seq_len), size=[batch_size, num], replace=True)

        # 4. 确定随机矩阵的源行索引 (Source Row Indices)
        # 逻辑：RandomMatrix行号 = 类别索引 * 20 + SNR偏移
        # 确保 Y_train 和 Z_train 转为 numpy 进行计算
        y_numpy = Y_train.cpu().numpy().astype(int)
        z_numpy = Z_train.cpu().numpy()

        # 计算当前样本对应的 SNR 索引 (0~19)
        # 公式: (-20 + 20)/2 = 0, (18 + 20)/2 = 19
        current_snr_idx = ((z_numpy + 20) / 2).astype(int)

        if low_snr:
            # 【Low SNR Mode】: 混合更低质量或同等质量的信号
            # 对每个样本，在 [0, current_snr_idx] 范围内随机选一个 SNR 索引
            # 这样做的目的是模拟该类别在不同信道条件下的特征
            offsets = np.array([np.random.randint(0, s + 1) for s in current_snr_idx])
            random_matrix_row_idxs = y_numpy * snr_len + offsets
        else:
            # 【Strict Mode】: 严格匹配当前 SNR
            random_matrix_row_idxs = y_numpy * snr_len + current_snr_idx

        # 5. 确定随机矩阵的源列索引 (Source Col Indices)
        # 从 RandomMatrix (长度6400) 中随机选 num 个点
        random_matrix_col_idxs = np.random.choice(range(self.random_matrix.shape[2]), size=[batch_size, num],
                                                  replace=True)

        # 6. 执行自适应替换 (Adaptive Replacement)
        # 使用循环处理以确保每个样本的缩放系数是独立的、准确的
        for i in range(batch_size):
            # A. 获取当前原始信号的能量水平 (Standard Deviation)
            # 加上 1e-9 防止全0信号导致除零错误
            # 这里的 std 代表了当前信号的平均幅度
            original_std = torch.std(X_train[i]) + 1e-9

            # B. 取出要注入的纯净片段 [2, num]
            rm_row = random_matrix_row_idxs[i]
            rm_cols = random_matrix_col_idxs[i]

            # 从 Random Matrix 提取数据
            clean_patch = self.random_matrix[rm_row, :, rm_cols]

            # C. 计算纯净片段的能量
            clean_std = torch.std(clean_patch) + 1e-9

            # D. 计算缩放因子 (Ratio)
            # 我们希望注入信号的幅度与原始信号一致，而不是像之前那样能量爆炸
            scale_factor = original_std / clean_std

            # E. 应用缩放
            clean_patch_scaled = clean_patch * scale_factor

            # F. 执行替换
            target_cols = res_index_col[i]
            res[i, :, target_cols] = clean_patch_scaled

        return res


    def analyze_inter_class_separation(self, task_id):
        """
        计算并打印/保存类间分离度，用于证明空间压缩假设 (Evidence B)
        """
        self._network.eval()
        with torch.no_grad():
            # 1. 获取当前所有原型
            if isinstance(self._network, nn.DataParallel):
                prototypes = self._network.module.fc.weight.data
            else:
                prototypes = self._network.fc.weight.data

            num_protos = prototypes.shape[0]

            # 2. 计算距离矩阵 (使用 Cosine Distance = 1 - Cosine Similarity)
            # 距离范围 [0, 2], 0表示重叠, 2表示方向相反
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            sim_matrix = torch.mm(proto_norm, proto_norm.t())
            dist_matrix = 1.0 - sim_matrix

            # 将对角线(自己到自己)设为无穷大，避免干扰最小值计算
            dist_matrix.fill_diagonal_(float('inf'))

            # 3. 统计分析
            # min_dists: 每个类别距离它最近邻居的距离
            min_dists, nearest_idx = torch.min(dist_matrix, dim=1)

            print(f"\n--- [Evidence B Analysis] Task {task_id} ---")
            print(f"Average Minimum Inter-class Distance: {min_dists.mean().item():.4f}")

            # 重点关注旧类 (Old Classes) 的情况
            if self._known_classes > 0:
                old_class_min_dists = min_dists[:self._known_classes]
                new_class_min_dists = min_dists[self._known_classes:]

                print(
                    f"Old Classes ({0}-{self._known_classes - 1}) Avg Separation: {old_class_min_dists.mean().item():.4f}")
                print(
                    f"New Classes ({self._known_classes}-{num_protos - 1}) Avg Separation: {new_class_min_dists.mean().item():.4f}")

                # 可选：打印具体的旧类距离变化，看是否在减小
                # print(f"Old Class Min Distances: {old_class_min_dists.cpu().numpy()}")

            return min_dists.cpu().numpy()

    def generate_and_plot_confusion_matrix(self, test_loader):
        self._network.eval()  # 设置为评估模式
        all_preds = []
        all_targets = []

        with torch.no_grad():
            # 遍历测试集获取预测结果和真实标签
            for _, inputs, targets, snr in test_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                outputs = self._network(inputs)
                logits = outputs["logits"]
                _, preds = torch.max(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # 加上 normalize='true'，表示按真实的类别（行）进行归一化，得到的是召回率/准确率
        cm = confusion_matrix(all_targets, all_preds, normalize='true')

        # ================= 核心修改点：仅动态调整画布大小 =================
        # 基础大小为 10x8。当类别数超过约 16 类时，画布会自动按比例拉伸
        # 确保每个格子都有足够的物理空间来显示 3 位小数
        fig_width = max(10.0, self._total_classes * 0.6)
        fig_height = max(8.0, self._total_classes * 0.5)
        plt.figure(figsize=(fig_width, fig_height))
        # ==================================================================

        # 完全保留你原有的热力图绘制参数 (annot=True, fmt='.3f')
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=np.arange(self._total_classes),
                    yticklabels=np.arange(self._total_classes))

        plt.title(f'混淆矩阵 - 阶段 {self._cur_task} (类别 0-{self._total_classes - 1})',
                  fontsize=15)
        plt.ylabel('真实标签', fontsize=12)
        plt.xlabel('预测标签', fontsize=12)

        # 创建保存目录
        save_dir = 'results/confusion_matrices_3'
        os.makedirs(save_dir, exist_ok=True)

        # 保存图片
        save_path = os.path.join(save_dir, f'cm_acc_task_{self._cur_task}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        plt.close()
        logging.info(f"Task__ {self._cur_task} Confusion matrix (Accuracy) saved to {save_path}")


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
