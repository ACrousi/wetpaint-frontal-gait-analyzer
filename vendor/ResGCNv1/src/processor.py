import logging, torch, torch.nn.functional as F, numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import utils as U
from .initializer import Initializer


class Processor(Initializer):
    def _regression_pred_values(self, out):
        """Return (N,) prediction values for regression mode with clear shape checks."""
        if out.dim() == 1:
            return out
        if out.dim() == 2 and out.size(1) == 1:
            return out[:, 0]
        raise ValueError(
            f'Regression mode expects model output shape (N, 1), but got {tuple(out.shape)}. '
            'Please set dataset_args.<dataset>.num_class=1 and use_ldl=false for direct regression.'
        )

    def train(self, epoch):
        self.model.train()
        start_train_time = time()
        train_metric, num_sample = 0, 0
        epoch_losses = []
        train_iter = self.train_loader if self.no_progress_bar else tqdm(self.train_loader, dynamic_ncols=True)

        # Check if using LDL (regression task)
        use_ldl = getattr(self.args, 'use_ldl', False)
        task_mode = getattr(self.args, 'task_mode', None)
        is_regression = task_mode == 'regression'
        epoch_mae = [] if (use_ldl or is_regression) else None
        epoch_mse = [] if (use_ldl or is_regression) else None

        for num, batch in enumerate(train_iter):
            self.optimizer.zero_grad()

            # Unpack batch
            x, y, *rest = batch
            original_label = None
            gait_params = None
            names = None
            
            if len(rest) == 3:
                # (x, y, original_label, gait_params, name)
                original_label, gait_params, names = rest
            elif len(rest) == 2:
                # Could be (original_label, gait_params) or (original_label, name)
                if isinstance(rest[1], torch.Tensor):
                    original_label, gait_params = rest
                else:
                    original_label, names = rest
            elif len(rest) == 1:
                original_label = rest[0]
            
            if gait_params is not None:
                gait_params = gait_params.float().to(self.device)

            # Using GPU
            x = x.float().to(self.device)
            if use_ldl or is_regression:
                y = y.float().to(self.device)
            else:
                y = y.long().to(self.device)   # For classification, y is class indices

            # Calculating Output
            out, _ = self.model(x, gait_params)

            # Updating Weights
            if is_regression:
                pred_values = self._regression_pred_values(out)
                loss = self.loss_func(pred_values, y)
            else:
                loss = self.loss_func(out, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            # Calculating Metrics
            num_sample += x.size(0)
            if is_regression:
                # Direct regression: compare model output with original label
                pred_values = self._regression_pred_values(out)
                target_values = original_label.float().to(self.device) if original_label is not None else y
                batch_mae = torch.abs(pred_values - target_values).mean().item()
                batch_mse = ((pred_values - target_values) ** 2).mean().item()
                train_metric += torch.abs(pred_values - target_values).sum().item()
                epoch_mae.append(batch_mae)
                epoch_mse.append(batch_mse)
            elif use_ldl:
                # For LDL, calculate MAE on expectation values vs original labels
                pred_probs = F.softmax(out, dim=1)
                pred_expectation = torch.sum(pred_probs * self.bin_centers.to(self.device), dim=1)
                # Use original_label for MAE calculation (more accurate)
                if original_label is not None:
                    original_label_tensor = original_label.float().to(self.device)
                    batch_mae = torch.abs(pred_expectation - original_label_tensor).mean().item()
                    batch_mse = ((pred_expectation - original_label_tensor) ** 2).mean().item()
                    train_metric += torch.abs(pred_expectation - original_label_tensor).sum().item()
                else:
                    # Fallback to target_expectation if original_label not available
                    target_expectation = torch.sum(y * self.bin_centers.to(self.device), dim=1)
                    batch_mae = torch.abs(pred_expectation - target_expectation).mean().item()
                    batch_mse = ((pred_expectation - target_expectation) ** 2).mean().item()
                    train_metric += torch.abs(pred_expectation - target_expectation).sum().item()
                epoch_mae.append(batch_mae)
                epoch_mse.append(batch_mse)
            else:
                # For classification, calculate accuracy
                reco_top1 = out.max(1)[1]
                train_metric += reco_top1.eq(y).sum().item()

            # Showing Progress
            lr = self.optimizer.param_groups[0]['lr']
            epoch_losses.append(loss.item())
            if self.scalar_writer:
                self.scalar_writer.add_scalar('learning_rate', lr, self.global_step)
                self.scalar_writer.add_scalar('train_loss', loss.item(), self.global_step)
            if self.no_progress_bar:
                logging.info('Epoch: {}/{}, Batch: {}/{}, Loss: {:.4f}, LR: {:.4f}'.format(
                    epoch+1, self.max_epoch, num+1, len(self.train_loader), loss.item(), lr
                ))
            else:
                train_iter.set_description('Loss: {:.4f}, LR: {:.4f}'.format(loss.item(), lr))

        # Showing Train Results
        if use_ldl or is_regression:
            train_metric /= num_sample  # MAE
            metric_name = 'MAE'
            metric_format = '{:.4f}'
        else:
            train_metric /= num_sample  # Accuracy
            metric_name = 'accuracy'
            metric_format = '{:.2%}'

        if self.scalar_writer:
            self.scalar_writer.add_scalar('train_metric', train_metric, self.global_step)
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        self.train_losses.append(avg_train_loss)
        if (use_ldl or is_regression) and epoch_mae:
            avg_train_mae = sum(epoch_mae) / len(epoch_mae)
            self.train_mae_values.append(avg_train_mae)
            avg_train_mse = sum(epoch_mse) / len(epoch_mse)
            self.train_mse_values.append(avg_train_mse)
        logging.info('Epoch: {}/{}, Training {}: {}, LR: {:.6f}, Training time: {:.2f}s'.format(
            epoch+1, self.max_epoch, metric_name, metric_format.format(train_metric), lr, time()-start_train_time
        ))
        logging.info('')

    def eval(self):
        self.model.eval()
        start_eval_time = time()
        with torch.no_grad():
            # Check if using LDL (regression task)
            use_ldl = getattr(self.args, 'use_ldl', False)
            task_mode = getattr(self.args, 'task_mode', None)
            is_regression = task_mode == 'regression'

            if use_ldl or is_regression:
                eval_metric = 0  # MAE for regression
                eval_mse = 0  # MSE for regression
                metric_name = 'MAE'
                # Collect predictions and targets for ranking correlations
                all_pred_expectations = []
                all_target_labels = []
            else:
                acc_top1, acc_top5 = 0, 0
                cm = np.zeros((self.num_class, self.num_class))
                metric_name = 'accuracy'

            num_sample, eval_loss = 0, []
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, batch in enumerate(eval_iter):

                # Unpack batch
                x, y, *rest = batch
                original_label = None
                gait_params = None
                names = None
                
                if len(rest) == 3:
                    original_label, gait_params, names = rest
                elif len(rest) == 2:
                    if isinstance(rest[1], torch.Tensor):
                        original_label, gait_params = rest
                    else:
                        original_label, names = rest
                elif len(rest) == 1:
                    original_label = rest[0]

                if gait_params is not None:
                    gait_params = gait_params.float().to(self.device)

                # Using GPU
                x = x.float().to(self.device)
                if use_ldl or is_regression:
                    y = y.float().to(self.device)
                else:
                    y = y.long().to(self.device)   # For classification, y is class indices

                # Calculating Output
                out, _ = self.model(x, gait_params)

                # Getting Loss
                if is_regression:
                    pred_values = self._regression_pred_values(out)
                    loss = self.loss_func(pred_values, y)
                else:
                    loss = self.loss_func(out, y)
                eval_loss.append(loss.item())

                # Calculating Metrics
                num_sample += x.size(0)
                if is_regression:
                    # Direct regression: compare model output with original label
                    pred_values = self._regression_pred_values(out)
                    target_values = original_label.float().to(self.device) if original_label is not None else y
                    eval_metric += torch.abs(pred_values - target_values).sum().item()
                    eval_mse += ((pred_values - target_values) ** 2).sum().item()
                    all_pred_expectations.extend(pred_values.cpu().numpy().tolist())
                    all_target_labels.extend(target_values.cpu().numpy().tolist())

                    for i in range(out.size(0)):
                        true_label = original_label[i].item() if original_label is not None else y[i].item()
                        print(f"Sample {num * self.eval_batch_size + i}: pred = {pred_values[i].item():.2f}, true = {true_label:.2f}")
                elif use_ldl:
                    # For LDL, calculate MAE on expectation values vs original labels
                    pred_probs = F.softmax(out, dim=1)
                    pred_expectation = torch.sum(pred_probs * self.bin_centers.to(self.device), dim=1)
                    # Use original_label for MAE calculation (more accurate)
                    if original_label is not None:
                        original_label_tensor = original_label.float().to(self.device)
                        eval_metric += torch.abs(pred_expectation - original_label_tensor).sum().item()
                        eval_mse += ((pred_expectation - original_label_tensor) ** 2).sum().item()
                        # Collect for ranking correlations
                        all_pred_expectations.extend(pred_expectation.cpu().numpy().tolist())
                        all_target_labels.extend(original_label.cpu().numpy().tolist())
                    else:
                        # Fallback to target_expectation if original_label not available
                        target_expectation = torch.sum(y * self.bin_centers.to(self.device), dim=1)
                        eval_metric += torch.abs(pred_expectation - target_expectation).sum().item()
                        eval_mse += ((pred_expectation - target_expectation) ** 2).sum().item()
                        # Collect for ranking correlations
                        all_pred_expectations.extend(pred_expectation.cpu().numpy().tolist())
                        all_target_labels.extend(target_expectation.cpu().numpy().tolist())

                    # Calculate target expectation from label distribution (true_exp)
                    target_expectation_from_dist = torch.sum(y * self.bin_centers.to(self.device), dim=1)

                    # Print pred_probs and expectation for each sample
                    for i in range(out.size(0)):
                        true_label = original_label[i].item() if original_label is not None else None
                        true_exp = target_expectation_from_dist[i].cpu().numpy()
                        pred_bin_index = torch.argmax(pred_probs[i]).cpu().numpy()
                        pred_label = self.bin_centers[pred_bin_index]
                        print(f"Sample {num * self.eval_batch_size + i}: pred_probs = {pred_probs[i].cpu().numpy()}, pred_exp = {pred_expectation[i].cpu().numpy()}, true_exp = {true_exp}, true_label = {true_label}")
                else:
                    # For classification, calculate accuracy
                    reco_top1 = out.max(1)[1]
                    acc_top1 += reco_top1.eq(y).sum().item()
                    reco_top5 = torch.topk(out,5)[1]
                    acc_top5 += sum([y[n] in reco_top5[n,:] for n in range(x.size(0))])

                    # Calculating Confusion Matrix
                    for i in range(x.size(0)):
                        cm[y[i], reco_top1[i]] += 1

                # Showing Progress
                if self.no_progress_bar and self.args.evaluate:
                    logging.info('Batch: {}/{}'.format(num+1, len(self.eval_loader)))

        # Showing Evaluating Results
        eval_loss = sum(eval_loss) / len(eval_loss)
        self.val_losses.append(eval_loss)
        eval_time = time() - start_eval_time
        eval_speed = len(self.eval_loader) * self.eval_batch_size / eval_time / len(self.args.gpus)

        if use_ldl or is_regression:
            eval_metric /= num_sample  # MAE
            eval_mse /= num_sample  # MSE
            self.val_mae_values.append(eval_metric)
            self.val_mse_values.append(eval_mse)
            
            # Calculate Spearman and Kendall ranking correlations
            spearman_corr, spearman_p = spearmanr(all_target_labels, all_pred_expectations)
            kendall_corr, kendall_p = kendalltau(all_target_labels, all_pred_expectations)
            
            logging.info('MAE: {:.4f}, MSE: {:.4f}, Mean loss:{:.4f}'.format(eval_metric, eval_mse, eval_loss))
            logging.info('Spearman ρ: {:.4f} (p={:.4e}), Kendall τ: {:.4f} (p={:.4e})'.format(
                spearman_corr, spearman_p, kendall_corr, kendall_p
            ))
            if self.scalar_writer:
                self.scalar_writer.add_scalar('eval_mae', eval_metric, self.global_step)
                self.scalar_writer.add_scalar('eval_mse', eval_mse, self.global_step)
                self.scalar_writer.add_scalar('eval_spearman', spearman_corr, self.global_step)
                self.scalar_writer.add_scalar('eval_kendall', kendall_corr, self.global_step)
        else:
            acc_top1 /= num_sample
            acc_top5 /= num_sample
            logging.info('Top-1 accuracy: {:.2%}, Top-5 accuracy: {:.2%}, Mean loss:{:.4f}'.format(
                acc_top1, acc_top5, eval_loss
            ))
            if self.scalar_writer:
                self.scalar_writer.add_scalar('eval_acc', acc_top1, self.global_step)

        logging.info('Evaluating time: {:.2f}s, Speed: {:.2f} sequnces/(second*GPU)'.format(
            eval_time, eval_speed
        ))
        logging.info('')
        if self.scalar_writer:
            self.scalar_writer.add_scalar('eval_loss', eval_loss, self.global_step)

        torch.cuda.empty_cache()
        if use_ldl or is_regression:
            return eval_metric, eval_mse, spearman_corr, None  # Return MAE, MSE, Spearman, cm
        else:
            return acc_top1, acc_top5, cm

    def start(self):
        start_time = time()
        if self.args.evaluate:
            if self.args.debug:
                logging.warning('Warning: Using debug setting now!')
                logging.info('')

            # Loading Evaluating Model
            logging.info('Loading evaluating model ...')
            checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
            if checkpoint:
                self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Successful!')
            logging.info('')

            # Evaluating
            logging.info('Starting evaluating ...')
            self.eval()
            logging.info('Finish evaluating!')

        else:
            # Resuming
            start_epoch = 0
            use_ldl = getattr(self.args, 'use_ldl', False)
            task_mode = getattr(self.args, 'task_mode', None)
            is_regression = task_mode == 'regression'
            # Get best_metric_criterion from config: 'mae', 'mse', or 'spearman'
            best_metric_criterion = getattr(self.args, 'best_metric_criterion', 'mae')
            if use_ldl or is_regression:
                # Initialize best_state with MAE, MSE, and Spearman
                best_state = {'mae': float('inf'), 'mse': float('inf'), 'spearman': -float('inf'), 'cm': None}
                best_metric_key = best_metric_criterion  # 'mae', 'mse', or 'spearman'
                logging.info(f'Using best model criterion: {best_metric_criterion}')
            else:
                best_state = {'acc_top1':0, 'acc_top5':0, 'cm':0}
                best_metric_key = 'acc_top1'
            if self.args.resume:
                logging.info('Loading checkpoint ...')
                checkpoint = U.load_checkpoint(self.args.work_dir)
                self.model.module.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
                best_state.update(checkpoint['best_state'])
                self.global_step = start_epoch * len(self.train_loader)
                logging.info('Start epoch: {}'.format(start_epoch+1))
                logging.info('Best accuracy: {:.2%}'.format(best_state['acc_top1']))
                logging.info('Successful!')
                logging.info('')

            # Initialize early stopping counter
            self.patience_count = 0

            # Training
            logging.info('Starting training ...')
            for epoch in range(start_epoch, self.max_epoch):

                # Training
                self.train(epoch)

                # Evaluating
                is_best = False
                if (epoch+1) % self.eval_interval(epoch) == 0:
                    logging.info('Evaluating for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                    eval_mae, eval_mse, eval_spearman, cm = self.eval()
                    if use_ldl or is_regression:
                        # Update best_state with current metrics
                        current_mae = eval_mae
                        current_mse = eval_mse
                        current_spearman = eval_spearman
                        
                        # Determine if this is the best model based on criterion
                        if best_metric_criterion == 'spearman':
                            # Higher Spearman is better
                            if current_spearman > best_state['spearman']:
                                is_best = True
                                best_state.update({'mae': current_mae, 'mse': current_mse, 'spearman': current_spearman, 'cm': cm})
                            best_metric_name = 'Spearman'
                        elif best_metric_criterion == 'mse':
                            # Lower MSE is better
                            if current_mse < best_state['mse']:
                                is_best = True
                                best_state.update({'mae': current_mae, 'mse': current_mse, 'spearman': current_spearman, 'cm': cm})
                            best_metric_name = 'MSE'
                        else:  # 'mae'
                            # Lower MAE is better
                            if current_mae < best_state['mae']:
                                is_best = True
                                best_state.update({'mae': current_mae, 'mse': current_mse, 'spearman': current_spearman, 'cm': cm})
                            best_metric_name = 'MAE'
                    else:
                        current_metric = eval_mae  # acc_top1
                        best_metric_key = 'acc_top1'
                        best_metric_name = 'top-1 accuracy'
                        # For classification, higher accuracy is better
                        if current_metric > best_state[best_metric_key]:
                            is_best = True
                            best_state.update({best_metric_key: current_metric, 'acc_top5': eval_spearman, 'cm': cm})

                # Saving Model
                logging.info('Saving model for epoch {}/{} ...'.format(epoch+1, self.max_epoch))
                save_interval = getattr(self.args, 'save_interval', 0)
                U.save_checkpoint(
                    self.model.module.state_dict(), self.optimizer.state_dict(), self.scheduler.state_dict(),
                    epoch+1, best_state, is_best, self.args.work_dir, self.save_dir, self.model_name,
                    save_interval=save_interval
                )
                if use_ldl or is_regression:
                    if best_metric_criterion == 'spearman':
                        logging.info('Best Spearman: {:.4f} (MAE: {:.4f}, MSE: {:.4f}), Total time: {}'.format(
                            best_state['spearman'], best_state['mae'], best_state['mse'], U.get_time(time()-start_time)
                        ))
                    elif best_metric_criterion == 'mse':
                        logging.info('Best MSE: {:.4f} (MAE: {:.4f}, Spearman: {:.4f}), Total time: {}'.format(
                            best_state['mse'], best_state['mae'], best_state['spearman'], U.get_time(time()-start_time)
                        ))
                    else:
                        logging.info('Best MAE: {:.4f} (MSE: {:.4f}, Spearman: {:.4f}), Total time: {}'.format(
                            best_state['mae'], best_state['mse'], best_state['spearman'], U.get_time(time()-start_time)
                        ))
                else:
                    logging.info('Best top-1 accuracy: {:.2%}, Total time: {}'.format(
                        best_state['acc_top1'], U.get_time(time()-start_time)
                    ))
                    logging.info('')
                self.plot_losses()
                logging.info('')

                # Early Stopping
                early_stop_patience = getattr(self.args, 'early_stop_patience', float('inf'))
                if is_best:
                    self.patience_count = 0
                else:
                    self.patience_count += 1
                    logging.info(f'EarlyStopping counter: {self.patience_count} out of {early_stop_patience}')
                    if self.patience_count >= early_stop_patience:
                        logging.info('Early stopping trigger!')
                        break

            # ====== K-Fold: 訓練結束後保存 fold 結果（用於論文報告） ======
            fold_idx = getattr(self.args, 'fold_idx', -1)
            if fold_idx >= 0 and (use_ldl or is_regression):
                import json, os

                # 保存 best metrics
                fold_results = {
                    'fold_idx': fold_idx,
                    'best_mae': best_state.get('mae', None),
                    'best_mse': best_state.get('mse', None),
                    'best_spearman': best_state.get('spearman', None),
                    'best_metric_criterion': best_metric_criterion,
                    'total_epochs': epoch + 1,
                }
                results_path = os.path.join(self.save_dir, 'fold_results.json')
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump(fold_results, f, ensure_ascii=False, indent=2)
                logging.info(f'Fold {fold_idx} results saved to: {results_path}')

                # 載入 best model 並執行最終 eval，保存每個 sample 的預測值
                logging.info(f'Loading best model for fold {fold_idx} final predictions ...')
                # 直接從 save_dir 載入（本次訓練剛存好的 best model）
                best_model_path = os.path.join(self.save_dir, f'{self.model_name}.pth.tar')
                if os.path.exists(best_model_path):
                    checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'), weights_only=False)
                    self.model.module.load_state_dict(checkpoint['model'])
                    self.model.eval()
                    all_preds, all_targets, all_names = [], [], []
                    with torch.no_grad():
                        for batch in self.eval_loader:
                            x, y, *rest = batch
                            original_label = None
                            gait_params = None
                            names = None
                            if len(rest) == 3:
                                original_label, gait_params, names = rest
                            elif len(rest) == 2:
                                if isinstance(rest[1], torch.Tensor):
                                    original_label, gait_params = rest
                                else:
                                    original_label, names = rest
                            elif len(rest) == 1:
                                original_label = rest[0]

                            if gait_params is not None:
                                gait_params = gait_params.float().to(self.device)
                            x = x.float().to(self.device)
                            out, _ = self.model(x, gait_params)

                            if is_regression:
                                pred_values = self._regression_pred_values(out).cpu().numpy().tolist()
                                target_values = (original_label.numpy().tolist()
                                                 if original_label is not None
                                                 else y.numpy().tolist())
                            else:
                                # LDL: compute expectation
                                pred_probs = torch.nn.functional.softmax(out, dim=1)
                                pred_values = torch.sum(
                                    pred_probs * self.bin_centers.to(self.device), dim=1
                                ).cpu().numpy().tolist()
                                target_values = (original_label.numpy().tolist()
                                                 if original_label is not None
                                                 else torch.sum(
                                                     y.float().to(self.device) * self.bin_centers.to(self.device), dim=1
                                                 ).cpu().numpy().tolist())

                            all_preds.extend(pred_values)
                            all_targets.extend(target_values)
                            if names is not None:
                                all_names.extend(names)

                    # 保存每個 sample 的預測值（用於論文畫圖）
                    predictions = {
                        'fold_idx': fold_idx,
                        'predictions': all_preds,
                        'targets': all_targets,
                        'sample_names': all_names if all_names else None,
                    }
                    pred_path = os.path.join(self.save_dir, 'fold_predictions.json')
                    with open(pred_path, 'w', encoding='utf-8') as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)
                    logging.info(f'Fold {fold_idx} predictions ({len(all_preds)} samples) saved to: {pred_path}')
                else:
                    logging.warning(f'Best model not found at {best_model_path}, skipping fold predictions.')

    def extract(self):
        logging.info('Starting extracting ...')
        if self.args.debug:
            logging.warning('Warning: Using debug setting now!')
            logging.info('')

        # Check if using LDL or regression
        use_ldl = getattr(self.args, 'use_ldl', False)
        task_mode = getattr(self.args, 'task_mode', None)
        is_regression = task_mode == 'regression'

        # Loading Model
        logging.info('Loading evaluating model ...')
        checkpoint = U.load_checkpoint(self.args.work_dir, self.model_name)
        cm = checkpoint['best_state']['cm']
        self.model.module.load_state_dict(checkpoint['model'])
        logging.info('Successful!')
        logging.info('')

        # Initialize lists to store all data
        all_data, all_labels, all_names, all_out, all_features = [], [], [], [], []
        all_locations = []
        all_class_labels = []
        if use_ldl or is_regression:
            all_pred_expectations = []
            all_target_expectations = []

        # Processing all batches
        self.model.eval()
        with torch.no_grad():
            eval_iter = self.eval_loader if self.no_progress_bar else tqdm(self.eval_loader, dynamic_ncols=True)
            for num, batch in enumerate(eval_iter):
                # Unpack batch: x, y, original_label, [gait_params,] names
                x, y, *rest = batch
                original_label = None
                gait_params = None
                names = None
                
                if len(rest) == 3:
                    original_label, gait_params, names = rest
                elif len(rest) == 2:
                    if isinstance(rest[1], torch.Tensor):
                        original_label, gait_params = rest
                    else:
                        original_label, names = rest
                elif len(rest) == 1:
                    original_label = rest[0]
                    
                gait_params = gait_params.float().to(self.device) if gait_params is not None else None

                # Using GPU
                x = x.float().to(self.device)
                if use_ldl or is_regression:
                    y = y.float().to(self.device)
                else:
                    y = y.long().to(self.device)

                # Calculating Output
                out, feature = self.model(x, gait_params)

                # Processing Data
                data = x.detach().cpu().numpy()
                label = y.detach().cpu().numpy()
                if is_regression:
                    out_processed = out.detach().cpu().numpy()
                else:
                    out_processed = torch.nn.functional.softmax(out, dim=1).detach().cpu().numpy()
                feature_processed = feature.detach().cpu().numpy()

                # Get class labels and expectations
                if is_regression:
                    class_label = label
                    # Direct regression: pred = model output, target = original label
                    pred_expectation = self._regression_pred_values(out).detach().cpu().numpy()
                    target_expectation = original_label.cpu().numpy() if original_label is not None else label
                    all_pred_expectations.append(pred_expectation)
                    all_target_expectations.append(target_expectation)
                elif use_ldl:
                    class_label = np.argmax(label, axis=1)
                    # Calculate expectations
                    pred_probs = torch.nn.functional.softmax(out, dim=1)
                    pred_expectation = torch.sum(pred_probs * self.bin_centers.to(self.device), dim=1).detach().cpu().numpy()
                    # Use original_label for target_expectations (more accurate)
                    if original_label is not None:
                        target_expectation = original_label.cpu().numpy()
                    else:
                        target_expectation = torch.sum(y * self.bin_centers.to(self.device), dim=1).detach().cpu().numpy()
                    all_pred_expectations.append(pred_expectation)
                    all_target_expectations.append(target_expectation)
                else:
                    class_label = label
                
                # Loading location data if available
                location = self.location_loader.load(names) if self.location_loader else []
                
                # Collecting all data
                all_data.append(data)
                all_labels.append(label)
                all_names.extend(names)
                all_out.append(out_processed)
                all_features.append(feature_processed)
                all_class_labels.append(class_label)
                if len(location) > 0:
                    all_locations.append(location)
                
                # Progress logging
                if self.no_progress_bar:
                    logging.info('Extracting batch: {}/{}'.format(num+1, len(self.eval_loader)))

        # Concatenate all batches
        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_out = np.concatenate(all_out, axis=0)
        all_features = np.concatenate(all_features, axis=0)
        all_class_labels = np.concatenate(all_class_labels, axis=0)
        if use_ldl or is_regression:
            all_pred_expectations = np.concatenate(all_pred_expectations, axis=0)
            all_target_expectations = np.concatenate(all_target_expectations, axis=0)
        if all_locations:
            all_locations = np.concatenate(all_locations, axis=0)
        else:
            all_locations = []

        # Get model weights
        weight = self.model.module.fcn.weight.squeeze().detach().cpu().numpy()

        logging.info('Extracted {} samples in total'.format(len(all_data)))

        # Saving Data
        if not self.args.debug:
            import os
            vis_dir = os.path.join(self.args.work_dir, 'visualization')
            U.create_folder(vis_dir)
            config_name = os.path.splitext(os.path.basename(self.args.config))[0]
            save_dict = {
                'data': all_data, 'label': all_labels, 'name': all_names, 'out': all_out, 'cm': cm,
                'feature': all_features, 'weight': weight, 'location': all_locations, 'class_label': all_class_labels
            }
            if use_ldl or is_regression:
                save_dict['pred_expectations'] = all_pred_expectations
                save_dict['target_expectations'] = all_target_expectations
            if use_ldl:
                save_dict['bin_centers'] = self.bin_centers.cpu().numpy()
            save_path = os.path.join(vis_dir, 'extraction_{}.npz'.format(config_name))
            np.savez(save_path, **save_dict)
            logging.info('Saved extraction to: {}'.format(save_path))
        logging.info('Finish extracting!')
        logging.info('')

    def predict(self, input_json_paths):
        """Predict mode: inference on input JSON files
        
        使用 ResGCNInference 封裝的推論流程，確保與訓練時資料處理一致
        
        Args:
            input_json_paths: List of JSON file paths to predict on
            
        Outputs:
            Prints JSON results to stdout for subprocess communication
        """
        import json
        import os
        from .inference import ResGCNInference
        
        logging.info('Starting prediction ...')
        
        if not input_json_paths:
            logging.error('No input JSON files provided!')
            return
        
        # 建立 ResGCNInference 實例
        pretrained_path = getattr(self.args, 'pretrained_path', '')
        inference = ResGCNInference.from_processor(
            self,
            checkpoint_path=pretrained_path if pretrained_path else None
        )
        
        # 執行批次預測
        predictions = inference.predict_batch(input_json_paths)
        
        # 轉換為字典格式
        results = [pred.to_dict() for pred in predictions]
        
        # Output results as JSON to stdout for subprocess communication
        print('===PREDICTION_RESULTS_START===')
        print(json.dumps(results, ensure_ascii=False, indent=2))
        print('===PREDICTION_RESULTS_END===')
        
        logging.info(f'Prediction completed! Processed {len(results)} files.')

    def plot_losses(self):
        """Plot training and validation loss curves, and MAE if available"""
        # Plot Loss curves
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label='Training Loss', color='blue')
        if self.val_losses:
            # Validation loss may not be recorded every epoch, adjust x-axis
            val_epochs = range(1, len(self.val_losses) + 1)
            plt.plot(val_epochs, self.val_losses, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('{}/loss_curves.png'.format(self.save_dir))
        plt.close()

        # Plot MAE curves if available
        if self.train_mae_values or self.val_mae_values:
            plt.figure(figsize=(10, 6))
            if self.train_mae_values:
                train_mae_epochs = range(1, len(self.train_mae_values) + 1)
                plt.plot(train_mae_epochs, self.train_mae_values, label='Training MAE', color='orange')
            if self.val_mae_values:
                val_mae_epochs = range(1, len(self.val_mae_values) + 1)
                plt.plot(val_mae_epochs, self.val_mae_values, label='Validation MAE', color='purple')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.title('Training and Validation MAE Curves')
            plt.legend()
            plt.grid(True)
            plt.savefig('{}/mae_curves.png'.format(self.save_dir))
            plt.close()

        # Plot MSE curves if available
        if self.train_mse_values or self.val_mse_values:
            plt.figure(figsize=(10, 6))
            if self.train_mse_values:
                train_mse_epochs = range(1, len(self.train_mse_values) + 1)
                plt.plot(train_mse_epochs, self.train_mse_values, label='Training MSE', color='green')
            if self.val_mse_values:
                val_mse_epochs = range(1, len(self.val_mse_values) + 1)
                plt.plot(val_mse_epochs, self.val_mse_values, label='Validation MSE', color='brown')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('Training and Validation MSE Curves')
            plt.legend()
            plt.grid(True)
            plt.savefig('{}/mse_curves.png'.format(self.save_dir))
            plt.close()
