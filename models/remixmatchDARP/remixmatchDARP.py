import os
import json
import contextlib
import numpy as np
from pathlib import Path
from copy import deepcopy
from sklearn.metrics import *
from imblearn.metrics import geometric_mean_score

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from utils.train_utils import ce_loss, EMA, Bn_Controller
from .darp_utils import estimate_pseudo, safe_opt_solver
from .remixmatch_utils import Get_Scalar, one_hot, mixup_one_target

np.set_printoptions(precision=3, suppress=True, formatter={"float": "{: 0.3f}".format})


class ReMixMatchDARP:
    def __init__(self, net_builder, num_classes, ema_m, T, lambda_u, w_match, num_eval_iter=1000, tb_log=None, logger=None):
        """
        class ReMixMatchDARP contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            lambda_u: ratio of unsupervised loss to supervised loss
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(ReMixMatchDARP, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        self.model = net_builder(num_classes=num_classes)
        self.ema_model = deepcopy(self.model)

        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.w_match = w_match  # weight of distribution matching
        self.lambda_u = lambda_u
        self.tb_log = tb_log

        self.optimizer = None
        self.scheduler = None

        self.it = 0

        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f"[!] data loader keys: {self.loader_dict.keys()}")

    def set_dset(self, dset_dict):
        self.dset_dict = dset_dict
        self.print_fn(f"[!] dataset keys: {self.loader_dict.keys()}")

    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, args):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()
        if args.resume == True:
            self.ema.load(self.ema_model)

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0

        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        # p(y) based on the labeled examples seen during training
        assert Path(args.ulb_dist_path).exists(), f"Can't Find the Distribution Estimation at {args.ulb_dist_path}"
        with open(args.ulb_dist_path, "r") as f:
            p_target = json.loads(f.read())
            p_target = torch.tensor(p_target["distribution"])
            p_target = p_target.cuda(args.gpu)
            ulb_dist = p_target * len(self.dset_dict["train_ulb"])
        self.print_fn(f"using p_target: {p_target.cpu().numpy()}")

        p_model = None

        # eval for once to verify if the checkpoint is loaded correctly
        if args.resume == True:
            eval_dict = self.evaluate(args=args)
            print(eval_dict)

        pseudo_labels = torch.ones((len(self.dset_dict["train_ulb"]), self.num_classes)) / self.num_classes
        pseudo_refine = torch.ones((len(self.dset_dict["train_ulb"]), self.num_classes)) / self.num_classes

        # x_ulb_s1_rot: rotated data, rot_v: rot angles
        for (_, x_lb, y_lb), (x_ulb_idx, x_ulb_w, x_ulb_s1, x_ulb_s2, x_ulb_s1_rot, rot_v) in zip(
            self.loader_dict["train_lb"], self.loader_dict["train_ulb"]
        ):

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break

            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            num_rot = x_ulb_s1_rot.shape[0]
            assert num_ulb == x_ulb_s1.shape[0]

            x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = (
                x_lb.cuda(args.gpu),
                x_ulb_w.cuda(args.gpu),
                x_ulb_s1.cuda(args.gpu),
                x_ulb_s2.cuda(args.gpu),
            )
            x_ulb_s1_rot = x_ulb_s1_rot.cuda(args.gpu)  # rot_image
            rot_v = rot_v.cuda(args.gpu)  # rot_label
            y_lb = y_lb.cuda(args.gpu)

            # inference and calculate sup/unsup losses
            with amp_cm():
                with torch.no_grad():
                    self.bn_controller.freeze_bn(self.model)
                    logits_x_ulb_w = self.model(x_ulb_w)[0]
                    self.bn_controller.unfreeze_bn(self.model)

                    # hyper-params for update
                    T = self.t_fn(self.it)

                    prob_x_ulb = torch.softmax(logits_x_ulb_w, dim=1)

                    if args.use_da:
                        # p^~_(y): moving average of p(y)
                        if p_model == None:
                            p_model = torch.mean(prob_x_ulb.detach(), dim=0)
                        else:
                            p_model = p_model * 0.999 + torch.mean(prob_x_ulb.detach(), dim=0) * 0.001
                        prob_x_ulb = prob_x_ulb * p_target / p_model
                        prob_x_ulb = prob_x_ulb / prob_x_ulb.sum(dim=-1, keepdim=True)

                    sharpen_prob_x_ulb = prob_x_ulb ** (1 / T)
                    sharpen_prob_x_ulb = (sharpen_prob_x_ulb / sharpen_prob_x_ulb.sum(dim=-1, keepdim=True)).detach()

                    # DARP
                    pseudo_labels[x_ulb_idx, :] = sharpen_prob_x_ulb.detach().cpu()

                    if self.it >= args.darp_warm:
                        if self.it % args.darp_update_iter == 0:
                            targets_u, weights_u = estimate_pseudo(ulb_dist, pseudo_labels, self.num_classes, args.darp_alpha)
                            scale_term = targets_u * weights_u.reshape(1, -1)
                            pseudo_orig_rm_noise = (pseudo_labels * scale_term + 1e-6) / (pseudo_labels * scale_term + 1e-6).sum(
                                dim=1, keepdim=True
                            )
                            pseudo_refine = safe_opt_solver(pseudo_orig_rm_noise, ulb_dist, args.darp_iter, args.darp_tol)

                        sharpen_prob_x_ulb = pseudo_refine[x_ulb_idx].detach().cuda(args.gpu)

                    # mix up
                    mixed_inputs = torch.cat((x_lb, x_ulb_s1, x_ulb_s2, x_ulb_w))
                    input_labels = torch.cat(
                        [one_hot(y_lb, args.num_classes, args.gpu), sharpen_prob_x_ulb, sharpen_prob_x_ulb, sharpen_prob_x_ulb], dim=0
                    )

                    mixed_x, mixed_y, _ = mixup_one_target(mixed_inputs, input_labels, args.gpu, args.alpha, is_bias=True)

                    # Interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                    mixed_x = list(torch.split(mixed_x, num_lb))
                    mixed_x = self.interleave(mixed_x, num_lb)

                logits = [self.model(mixed_x[0])[0]]

                self.bn_controller.freeze_bn(self.model)
                for ipt in mixed_x[1:]:
                    logits.append(self.model(ipt)[0])

                u1_logits = self.model(x_ulb_s1)[0]
                logits_rot = self.model(x_ulb_s1_rot)[1]
                logits = self.interleave(logits, num_lb)
                self.bn_controller.unfreeze_bn(self.model)

                logits_x = logits[0]
                logits_u = torch.cat(logits[1:])

                # calculate rot loss with w_rot
                rot_loss = ce_loss(logits_rot, rot_v, reduction="mean")
                rot_loss = rot_loss.mean()
                # sup loss
                sup_loss = ce_loss(logits_x, mixed_y[:num_lb], use_hard_labels=False)
                sup_loss = sup_loss.mean()
                # unsup_loss
                unsup_loss = ce_loss(logits_u, mixed_y[num_lb:], use_hard_labels=False)
                unsup_loss = unsup_loss.mean()
                # loss U1
                u1_loss = ce_loss(u1_logits, sharpen_prob_x_ulb, use_hard_labels=False)
                u1_loss = u1_loss.mean()
                # ramp for w_match
                w_match = args.w_match * float(np.clip(self.it / (args.warm_up * args.num_train_iter), 0.0, 1.0))
                w_kl = args.w_kl * float(np.clip(self.it / (args.warm_up * args.num_train_iter), 0.0, 1.0))

                total_loss = sup_loss + args.w_rot * rot_loss + w_kl * u1_loss + w_match * unsup_loss

            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()

            self.scheduler.step()
            self.ema.update()
            self.model.zero_grad()

            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            tb_dict["train/sup_loss"] = sup_loss.detach()
            tb_dict["train/unsup_loss"] = unsup_loss.detach()
            tb_dict["train/total_loss"] = total_loss.detach()
            tb_dict["lr"] = self.optimizer.param_groups[0]["lr"]
            tb_dict["train/prefecth_time"] = start_batch.elapsed_time(end_batch) / 1000.0
            tb_dict["train/run_time"] = start_run.elapsed_time(end_run) / 1000.0

            # Save model for each 10K steps
            if self.it % 10000 == 0:
                save_path = os.path.join(args.save_dir, args.save_name)
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    self.save_model("latest_model.pth", save_path)

            if self.it % self.num_eval_iter == 0:
                self.print_fn("evaluation:")
                eval_dict = self.evaluate(args=args)
                tb_dict.update(eval_dict)

                save_path = os.path.join(args.save_dir, args.save_name)

                if tb_dict["eval/top-1-acc"] > best_eval_acc:
                    best_eval_acc = tb_dict["eval/top-1-acc"]
                    best_it = self.it

                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters"
                )

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model("model_best.pth", save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)

            self.it += 1
            del tb_dict
            start_batch.record()

        eval_dict = self.evaluate(args=args)
        eval_dict.update({"eval/best_acc": best_eval_acc, "eval/best_it": best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict["eval"]
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits, y, reduction="mean")
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        geo_mean = geometric_mean_score(y_true, y_pred, correction=0.001)
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        F1 = f1_score(y_true, y_pred, average="macro")
        AUC = roc_auc_score(y_true, y_logits, multi_class="ovo")
        metrics = {}
        if self.num_classes <= 10:
            cf_mat = confusion_matrix(y_true, y_pred, normalize="true")
            self.print_fn("confusion matrix:\n" + np.array_str(cf_mat))
            classwise_acc = cf_mat.diagonal()
            metrics.update({f"eval_classes_acc/{c}": classwise_acc[c] for c in range(self.num_classes)})
        metrics.update(
            {
                "eval/loss": total_loss / total_num,
                "eval/top-1-acc": top1,
                "eval/top-5-acc": top5,
                "eval/balanced-acc": balanced_acc,
                "eval/geometric-mean": geo_mean,
                "eval/precision": precision,
                "eval/recall": recall,
                "eval/F1": F1,
                "eval/AUC": AUC,
            }
        )
        self.ema.restore()
        self.model.train()
        return metrics

    def save_model(self, save_name, save_path):
        if self.it < 100000:
            return
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = deepcopy(self.model)
        self.ema.restore()
        self.model.train()

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "it": self.it + 1,
                "ema_model": ema_model.state_dict(),
            },
            save_filename,
        )

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        checkpoint = torch.load(load_path)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.it = checkpoint["it"]
        self.ema_model.load_state_dict(checkpoint["ema_model"])
        self.print_fn("model loaded")

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
