
from model.trainer.base import Trainer 
from model.trainer.helpers import (get_dataloader, prepare_model, prepare_optimizer)
from edl_losses import select_edl_loss, relu_evidence
from model.utils import (pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval)

from sklearn.metrics import roc_auc_score, roc_curve
import torch 
import time 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import numpy as np 
import os.path as osp 
from tqdm import tqdm
from torch import nn 
import libmr

from model.utils.hsc_utils import extrap_mix, energy_diffusion, sample_far_ood
from model.losses.energy_loss import angle_compact_loss, energy_shaping_loss, kl_unknown_loss


cos = nn.CosineSimilarity()
from torch.autograd import Variable
label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse = False)

def get_onehot_encoder(y):
    le = label_encoder.fit(y)
    integer_encoded = le.transform(y).reshape(-1, 1)
    y_hot = onehot_encoder.fit_transform(integer_encoded)
    return y_hot

def calc_auroc(known_scores, unknown_scores):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    
    return auc_score

    



class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.val_loader, self.test_loader = get_dataloader(args)
        self.model, self.para_model = prepare_model(args)
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

        self.kappa = getattr(self.args, 'kappa', 10.0)
        self.open_margin = getattr(self.args, 'open_margin', 0.1)
        self.lambda_energy = getattr(self.args, 'lambda_energy', 1.0)
        self.lambda_edl = getattr(self.args, 'lambda_edl', 1.0)
        self.lambda_kl = getattr(self.args, 'lambda_kl', 1.0)
        self.diffusion_steps = getattr(self.args, 'diffusion_steps', 3)

        # 远域 OOD 可选：若你已经在 helpers 里构造了 ood_loader，可在 args 传入
        self.ood_loader = getattr(self, 'ood_loader', None)


    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.closed_way, dtype=torch.int16).repeat(args.query)
        label_hot = get_onehot_encoder(label.data.cpu().numpy())
        label_hot = torch.from_numpy(label_hot)
        label_aux = torch.arange(args.closed_way, dtype=torch.int8).repeat(args.shot + args.query)
        
        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)
        
        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()
            
        return label, label_aux, label_hot



    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # start FSL training 
        label, label_aux, label_hot = self.prepare_label()
        if torch.cuda.is_available():
            label_hot = label_hot.cuda(non_blocking=True)

        for epoch in range(1, args.max_epoch+1):
            self.train_epoch+=1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            
            tl1 = Averager()
            tl2 = Averager()
            ta = Averager()

            start_tm = time.time()
            for batch in self.train_loader:
                self.train_step+=1

                if torch.cuda.is_available():
                    data, gt_label = [_.cuda() for _ in batch]
                else:
                    data, gt_label = batch[0], batch[1]
                
                
                data_tm = time.time()
                self.dt.add(data_tm-start_tm)
                 # get saved centers
                # if args.open_loss:
                #     logits, open_logits, reg_logits = self.para_model(data)
                # else:

                #     logits, reg_logits = self.para_model(data)
                
                # if reg_logits is not None:
                #     if args.loss_type=='ce_loss':
                #         loss = F.cross_entropy(logits, label)
                #     elif args.loss_type=='edl_loss':
                #         loss = select_edl_loss(logits, label_hot, epoch, self.args.closed_way, self.args)
                
                # if args.open_loss:
                #     Q = torch.ones_like(open_logits)
                #     P = open_logits+1
                #     open_loss = (P*(P/Q).log()).sum(axis = 1).mean()
                #     total_loss = loss +args.balance * F.cross_entropy(reg_logits, label_aux)+args.open_loss_coeff*open_loss
                # else:
                #     total_loss = loss +args.balance * F.cross_entropy(reg_logits, label_aux)
                # hsc loss as follows:
                model_name = str(self.args.model_class).lower()

                if model_name in ['hsc', 'hscmodel']:
                    # ===== HSC branch =====
                    # 1) 前向：建议 HSCModel.forward 返回 dict
                    #    必须键：close_logits [Bq,C_close]，energies [Bq]，evidence [Bq,C_close]
                    #    可选键：unk_energies [Nu]（若 HSCModel 已在内部生成未知）、reg_logits（若仍用原reg头）
                    out = self.para_model(data)
                    if isinstance(out, dict):
                        close_logits = out['close_logits']
                        energies     = out['energies']           # [Bq]
                        evidence     = out['evidence']           # [Bq, C]
                        reg_logits   = out.get('reg_logits', None)
                        unk_energies = out.get('unk_energies', None)
                    elif isinstance(out, (list, tuple)) and len(out) >= 3:
                        close_logits, energies, evidence = out[0], out[1], out[2]
                        reg_logits   = out[3] if len(out) > 3 else None
                        unk_energies = out[4] if len(out) > 4 else None
                    else:
                        raise RuntimeError('HSCModel forward outputs not recognized, expect dict or tuple.')

                    # 2) 闭集角度紧致损失（基于 logits 的版本）
                    loss_ac = angle_compact_loss(
                        features=None, prototypes=None,
                        labels=label, margin=self.open_margin, kappa=self.kappa,
                        logits=close_logits
                    )

                    # 3) 能量整形：闭集->低能量，未知->高能量
                    e_target_closed  = 0.1
                    e_target_unknown = 0.9

                    loss_energy_closed = energy_shaping_loss(
                        energies, torch.full_like(energies, e_target_closed), reduction='mean'
                    )
                    if unk_energies is not None and unk_energies.numel() > 0:
                        loss_energy_unknown = energy_shaping_loss(
                            unk_energies, torch.full_like(unk_energies, e_target_unknown), reduction='mean'
                        )
                        loss_energy = 0.5 * (loss_energy_closed + loss_energy_unknown)
                    else:
                        # 若 HSCModel 暂未返回 unk_energies，也可以先只约束闭集能量
                        loss_energy = loss_energy_closed

                    # 4) 证据正则（最简：未知证据→均匀Dirichlet；若模型已返回 evidence_unknown 更好）
                    # 从 HSCModel.forward 返回的 out 中取未知证据
                    evidence_unknown = out.get('evidence_unknown', None)

                    if evidence_unknown is not None and evidence_unknown.numel() > 0:
                        loss_kl = kl_unknown_loss(
                            evidence=evidence_unknown,           # <—— 绑定真实未知证据
                            num_classes=close_logits.size(1),
                            reduction='mean'
                        )
                    else:
                        loss_kl = 0.0


                    # 5) （可选）保持原 reg 支路的平衡项
                    reg_term = 0.0
                    if reg_logits is not None:
                        reg_term = self.args.balance * F.cross_entropy(reg_logits, label_aux)

                    total_loss = loss_ac + self.lambda_energy * loss_energy + self.lambda_kl * loss_kl + reg_term

                    # 6) 指标
                    logits = close_logits  # 复用下方的 acc 计算

                else:
                    # ===== 原 FEAT / 其他模型分支保留 =====
                    if args.open_loss:
                        logits, open_logits, reg_logits = self.para_model(data)
                    else:
                        logits, reg_logits = self.para_model(data)

                    if reg_logits is not None:
                        if args.loss_type == 'ce_loss':
                            loss = F.cross_entropy(logits, label)
                        elif args.loss_type == 'edl_loss':
                            loss = select_edl_loss(logits, label_hot, epoch, self.args.closed_way, self.args)
                        else:
                            raise ValueError('Unknown loss_type: {}'.format(args.loss_type))
                    else:
                        # 极端情况下没有 reg_logits，给个兜底
                        loss = F.cross_entropy(logits, label)

                    if args.open_loss:
                        Q = torch.ones_like(open_logits)
                        P = open_logits + 1
                        open_loss = (P * (P / Q).log()).sum(dim=1).mean()
                        total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux) + args.open_loss_coeff * open_loss
                    else:
                        # 你原代码这里行末有一个多余的 '+' 会导致语法错误，修正如下：
                        total_loss = loss + args.balance * F.cross_entropy(reg_logits, label_aux)
                # hsc loss end


                tl2.add(total_loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                acc = count_acc(logits, label)
                #print("Accuracy is", acc, "loss is", total_loss)
                tl1.add(total_loss.item())
                ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)    

                # refresh start_tm
                start_tm = time.time()

            self.lr_scheduler.step()
            self.try_evaluate(epoch)
            print('ETA:{}/{}'.format(
                    self.timer.measure(),
                    self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')
        self.logger.dump()

   
    def get_snatcher_prob(self, logits, bproto, emb_dim, proto, query):
        snatch = []
        for j in range(logits.shape[0]):
            pproto = bproto.clone().detach()
            c = logits.argmax(1)[j]
            """Algorithm 1 Line 2"""
            pproto[0][c] = query.reshape(-1, emb_dim)[j]
            """Algorithm 1 Line 3"""
            pproto, _ = self.model.slf_attn(pproto, pproto, pproto)
            pdiff = (pproto-proto).pow(2).sum(-1).sum()/64.0
            """pdiff: d_SnaTCHer in Algorithm 1"""
            snatch.append(pdiff)
        return snatch


    def evaluate(self, data_loader):
        args = self.args
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 7))
        label = torch.arange(args.closed_way, dtype = torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        
        
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

       
        with torch.no_grad():
            for i, batch in enumerate(data_loader, 1):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data = batch[0]

                    model_name = str(self.args.model_class).lower()
                    if model_name in ['hsc', 'hscmodel']:
                        # ===== HSC evaluation branch =====
                        args = self.args
                        device = next(self.para_model.parameters()).device

                        # 1) 取实例特征与 episodic 索引（用 HSCModel 的 encoder/split_instances）
                        x = data.squeeze(0)  # [N,C,H,W]
                        instance_embs = self.para_model.encoder(x)                      # [N,D]
                        support_idx, query_idx = self.para_model.split_instances(x)     # 与训练同源

                        # 2) 根据 episodic 划分 support/query，并拆分已知/未知 query
                        support = instance_embs[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
                        query   = instance_embs[query_idx.view(-1)].view(  *(query_idx.shape   + (-1,)))
                        emb_dim = support.shape[-1]

                        support_close = support[:, :, :args.closed_way, :]                      # [B,shot,C,D]
                        bproto = support_close.mean(dim=1)                                      # [B=1,C,D]
                        proto  = F.normalize(bproto, dim=-1).squeeze(0)                         # [C,D]

                        kquery = query[:, :, :args.closed_way, :].contiguous()                  # [B,q,C,D]
                        uquery = query[:, :, args.closed_way:, :].contiguous()                  # [B,q_open,?,D] 可能为0

                        # 3) 闭集 logits（HSC 的 κ-scaled cosine + margin）
                        kfeat = kquery.reshape(-1, emb_dim)
                        kfeat_n = F.normalize(kfeat, dim=-1)
                        klogits = self.kappa * torch.matmul(kfeat_n, proto.t()) - self.open_margin  # [Bq*C,C]

                        # 4) 能量与证据：已知/未知分别过 head
                        k_energy = self.para_model.energy_head(kfeat).squeeze(-1)                # [Nk]
                        k_evid  = self.para_model.evidence_head(kfeat)                           # [Nk,C] (Softplus>=0)

                        if uquery.numel() > 0:
                            ufeat = uquery.reshape(-1, emb_dim)
                            u_energy = self.para_model.energy_head(ufeat).squeeze(-1)            # [Nu]
                            u_evid  = self.para_model.evidence_head(ufeat)                       # [Nu,C]
                        else:
                            u_energy = None
                            u_evid   = None

                        # 5) 指标：闭集 top-1
                        acc = count_acc(klogits, label)

                        # 6) 开集：能量 AUROC（已知低、未知高）
                        if u_energy is not None and u_energy.numel() > 0:
                            auroc_energy = calc_auroc(
                                k_energy.detach().cpu().numpy(),
                                u_energy.detach().cpu().numpy()
                            )
                        else:
                            auroc_energy = 0.0  # 若当前 episode 无 open queries

                        # 7) 不确定性（vacuity）AUROC（可选）
                        #   alpha = evidence + 1, S = sum(alpha), vacuity = C / S
                        C = args.closed_way
                        k_alpha = k_evid + 1.0
                        k_S = torch.sum(k_alpha, dim=1)
                        k_vacuity = (C / k_S).detach().cpu().numpy()

                        if u_evid is not None and u_evid.numel() > 0:
                            u_alpha = u_evid + 1.0
                            u_S = torch.sum(u_alpha, dim=1)
                            u_vacuity = (C / u_S).detach().cpu().numpy()
                            auroc_vacuity = calc_auroc(k_vacuity, u_vacuity)
                        else:
                            auroc_vacuity = 0.0

                        # 8) 记录（沿用原 record 布局：0=loss,1=acc,2=prob-auroc,3=dist-auroc,4=snatch-auroc,5=edl-auroc）
                        #   这里我们把第2列复用为 energy-AUROC，第5列复用为 vacuity-AUROC，其余置0
                        record[i-1, 0] = 0.0
                        record[i-1, 1] = acc
                        record[i-1, 2] = auroc_energy
                        record[i-1, 3] = 0.0
                        record[i-1, 4] = 0.0
                        record[i-1, 5] = auroc_vacuity

                    else:
                        # ===== 原 FEAT / 其他模型评估分支，保持不变 =====
                        [instance_embs, support_idx, query_idx]  = self.model(data)
                        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
                        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,)))
                        emb_dim = support.shape[-1]

                        support = support[:, :, :args.closed_way].contiguous()
                        bproto = support.mean(dim = 1)
                        proto = bproto

                        kquery = query[:, :, :args.closed_way].contiguous()
                        uquery = query[:, :, args.closed_way:].contiguous()
                        proto, _ = self.model.slf_attn(proto, proto, proto)

                        klogits = -(kquery.reshape(-1, 1, emb_dim)-proto).pow(2).sum(2)/64.0
                        ulogits = -(uquery.reshape(-1, 1, emb_dim)-proto).pow(2).sum(2)/64.0
                        loss = F.cross_entropy(klogits, label)
                        acc = count_acc(klogits, label)

                        known_prob = F.softmax(klogits, 1).max(1)[0]
                        unknown_prob = F.softmax(ulogits, 1).max(1)[0]
                        known_scores = (1-known_prob).cpu().detach().numpy()
                        unknown_scores = (1-unknown_prob).cpu().detach().numpy()
                        auroc = calc_auroc(known_scores, unknown_scores)

                        kdist = -(klogits.max(1)[0]).cpu().detach().numpy()
                        udist = -(ulogits.max(1)[0]).cpu().detach().numpy()
                        dist_auroc = calc_auroc(kdist, udist)

                        snatch_known = self.get_snatcher_prob(klogits, bproto, emb_dim, proto, kquery)
                        snatch_unknown = self.get_snatcher_prob(ulogits, bproto, emb_dim, proto, uquery)
                        pkdiff = torch.stack(snatch_known).cpu().detach().numpy()
                        pudiff = torch.stack(snatch_unknown).cpu().detach().numpy()
                        snatch_auroc = calc_auroc(pkdiff, pudiff)

                        k_evidence = relu_evidence(-1/klogits)
                        u_evidence = relu_evidence(-1/ulogits)
                        k_alpha = k_evidence+1
                        u_alpha = u_evidence+1
                        k_s = torch.sum(k_alpha, axis = 1)
                        u_s = torch.sum(u_alpha, axis = 1)
                        k_uncert = args.closed_way/k_s
                        u_uncert = args.closed_way/u_s
                        edl_auroc = calc_auroc(k_uncert.cpu().detach().numpy(), u_uncert.cpu().detach().numpy())

                        record[i-1, 0] = loss.item()
                        record[i-1, 1] = acc
                        record[i-1, 2] = auroc
                        record[i-1, 3] = dist_auroc
                        record[i-1, 4] = snatch_auroc
                        record[i-1, 5] = edl_auroc
                
               
        assert(i==record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        vaccm, vaccs = compute_confidence_interval(record[:, 1])
        vaucmp, vaucsp = compute_confidence_interval(record[:, 2])
        vaucmd, vaucsd = compute_confidence_interval(record[:, 3])
        vaucms, vaucss = compute_confidence_interval(record[:, 4])
        vaucmedl, vaucsedl = compute_confidence_interval(record[:, 5])

        return vl, vaccm, vaccs, vaucmp, vaucsp, vaucmd, vaucsd, vaucms, vaucss, vaucmedl, vaucsedl
    
    def evaluate_test(self):
        args = self.args 

        if args.open_loss:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+str(args.open_loss_coeff)+'_olf_'+self.args.loss_type+'_'+'max_acc'+'.pth'))['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+self.args.loss_type+'_max_acc'+'.pth'))['params'])
        
        self.model.eval()

        record = np.zeros((1000, 7))
        label = torch.arange(args.closed_way, dtype = torch.int16).repeat(args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
          
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader, 1):
                if torch.cuda.is_available():
                    data, classes, image_names = batch
                    classes = classes.cuda()
                    data = data.cuda()
                else:
                    data = batch[0]

                model_name = str(self.args.model_class).lower()
                if model_name in ['hsc', 'hscmodel']:
                    # ===== HSC test branch =====
                    args = self.args
                    device = next(self.para_model.parameters()).device
                    x = data.squeeze(0)
                    instance_embs = self.para_model.encoder(x)
                    support_idx, query_idx = self.para_model.split_instances(x)
                    support = instance_embs[support_idx.view(-1)].view(*(support_idx.shape + (-1,)))
                    query = instance_embs[query_idx.view(-1)].view(*(query_idx.shape + (-1,)))
                    emb_dim = support.shape[-1]
                    support_close = support[:, :, :args.closed_way, :]
                    bproto = support_close.mean(dim=1)
                    proto = F.normalize(bproto, dim=-1).squeeze(0)
                    kquery = query[:, :, :args.closed_way, :].contiguous()
                    uquery = query[:, :, args.closed_way:, :].contiguous()
                    kfeat = kquery.reshape(-1, emb_dim)
                    kfeat_n = F.normalize(kfeat, dim=-1)
                    klogits = self.kappa * torch.matmul(kfeat_n, proto.t()) - self.open_margin
                    k_energy = self.para_model.energy_head(kfeat).squeeze(-1)
                    k_evid = self.para_model.evidence_head(kfeat)
                    if uquery.numel() > 0:
                        ufeat = uquery.reshape(-1, emb_dim)
                        u_energy = self.para_model.energy_head(ufeat).squeeze(-1)
                        u_evid = self.para_model.evidence_head(ufeat)
                    else:
                        u_energy = None
                        u_evid = None
                    acc = count_acc(klogits, label)
                    if u_energy is not None and u_energy.numel() > 0:
                        auroc_energy = calc_auroc(
                            k_energy.detach().cpu().numpy(),
                            u_energy.detach().cpu().numpy())
                    else:
                        auroc_energy = 0.0
                    C = args.closed_way
                    k_alpha = k_evid + 1.0
                    k_S = torch.sum(k_alpha, dim=1)
                    k_vacuity = (C / k_S).detach().cpu().numpy()
                    if u_evid is not None and u_evid.numel() > 0:
                        u_alpha = u_evid + 1.0
                        u_S = torch.sum(u_alpha, dim=1)
                        u_vacuity = (C / u_S).detach().cpu().numpy()
                        auroc_vacuity = calc_auroc(k_vacuity, u_vacuity)
                    else:
                        auroc_vacuity = 0.0
                    record[i-1, 0] = 0.0
                    record[i-1, 1] = acc
                    record[i-1, 2] = auroc_energy
                    record[i-1, 3] = 0.0
                    record[i-1, 4] = 0.0
                    record[i-1, 5] = auroc_vacuity
                    record[i-1, 6] = 0.0
                else:
                    # ===== 原 FEAT 分支保留不变 =====
                    [instance_embs, support_idx, query_idx] = self.model(data)
                    support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
                    query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
                    emb_dim = support.shape[-1]
                    support = support[:, :, :args.closed_way].contiguous()
                    bproto = support.mean(dim=1)
                    proto = bproto
                    kquery = query[:, :, :args.closed_way].contiguous()
                    uquery = query[:, :, args.closed_way:].contiguous()
                    proto, _ = self.model.slf_attn(proto, proto, proto)
                    klogits = -(kquery.reshape(-1, 1, emb_dim)-proto).pow(2).sum(2)/64.0
                    ulogits = -(uquery.reshape(-1, 1, emb_dim)-proto).pow(2).sum(2)/64.0
                    loss = F.cross_entropy(klogits, label)
                    acc = count_acc(klogits, label)
                    known_prob = F.softmax(klogits, 1).max(1)[0]
                    unknown_prob = F.softmax(ulogits, 1).max(1)[0]
                    known_scores = 1-known_prob.cpu().detach().numpy()
                    unknown_scores = 1-unknown_prob.cpu().detach().numpy()
                    auroc = calc_auroc(known_scores, unknown_scores)
                    kdist = -(klogits.max(1)[0]).cpu().detach().numpy()
                    udist = -(ulogits.max(1)[0]).cpu().detach().numpy()
                    dist_auroc = calc_auroc(kdist, udist)
                    snatch_known = self.get_snatcher_prob(klogits, bproto, emb_dim, proto, kquery)
                    snatch_unknown = self.get_snatcher_prob(ulogits, bproto, emb_dim, proto, uquery)
                    pkdiff = torch.stack(snatch_known).cpu().detach().numpy()
                    pudiff = torch.stack(snatch_unknown).cpu().detach().numpy()
                    snatch_auroc = calc_auroc(pkdiff, pudiff)
                    k_evidence = relu_evidence(-1/klogits)
                    u_evidence = relu_evidence(-1/ulogits)
                    k_alpha = k_evidence+1
                    u_alpha = u_evidence+1
                    k_s = torch.sum(k_alpha, axis=1)
                    u_s = torch.sum(u_alpha, axis=1)
                    k_uncert = args.closed_way/k_s
                    u_uncert = args.closed_way/u_s
                    edl_auroc = calc_auroc(k_uncert.cpu().detach().numpy(), u_uncert.cpu().detach().numpy())
                    record[i-1, 0] = loss.item()
                    record[i-1, 1] = acc
                    record[i-1, 2] = auroc
                    record[i-1, 3] = dist_auroc
                    record[i-1, 4] = snatch_auroc
                    record[i-1, 5] = edl_auroc
                    record[i-1, 6] = 0.0
                
                
        assert(i==record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        vaccm, vaccs = compute_confidence_interval(record[:, 1])
        vaucmp, vaucsp = compute_confidence_interval(record[:, 2])
        vaucmd, vaucsd = compute_confidence_interval(record[:, 3])
        vaucms, vaucss = compute_confidence_interval(record[:, 4])
        vaucmedl, vaucsedl = compute_confidence_interval(record[:, 5])
        vaucmevr, vaucsevr = compute_confidence_interval(record[:, 6])
        self.trlog['test_loss'] = vl 
        self.trlog['test_acc'] = float(vaccm)
        self.trlog['test_acc_interval'] = float(vaccs)
        self.trlog['test_auc_prob']= float(vaucmp)
        self.trlog['test_auc_prob_interval']= float(vaucsp)
        self.trlog['test_auc_dist']=float(vaucmd)
        self.trlog['test_auc_dist_interval']=float(vaucsd)
        self.trlog['test_auc_snatcher']= float(vaucms)
        self.trlog['test_auc_snatcher_interval']= float(vaucss)
        self.trlog['test_auc_edl']=float(vaucmedl)
        self.trlog['test_auc_edl_interval']=float(vaucsedl)
        self.trlog['test_auc_evr']=float(vaucmevr)
        self.trlog['test_auc_evr_interval']=float(vaucsevr)


    def final_record(self):
        # save the best performance in a txt file
        if self.args.open_loss:
            save_path = osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+str(self.args.open_loss_coeff)+'_olf_'+self.args.loss_type+'_openset_'+'{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval']))
        else:
            save_path = osp.join(self.args.save_path, str(self.args.shot)+'_s_'+str(self.args.run)+'_r_'+str(self.args.balance)+'_bal_'+self.args.loss_type+'_'+'{}+{}'.format(self.trlog['test_acc'], self.trlog['test_acc_interval']))

        with open(save_path , 'w') as f:
            f.write('best epoch {}, best val acc={:.4f} + {:.4f}\n'.format(
                self.trlog['max_acc_epoch'],
                self.trlog['max_acc'],
                self.trlog['max_acc_interval']))
            f.write('Test acc={:.4f} + {:.4f}\n'.format(
                self.trlog['test_acc'],
                self.trlog['test_acc_interval']))   


  
    