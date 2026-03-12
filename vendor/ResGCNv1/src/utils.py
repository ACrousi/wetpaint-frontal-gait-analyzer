import os, sys, shutil, logging, json, torch
from time import time, strftime, localtime


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def set_logging(save_dir):
    log_format = '[ %(asctime)s ] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
    handler = logging.FileHandler('{}/log.txt'.format(save_dir), mode='w', encoding='UTF-8')
    handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(handler)


def get_time(total_time):
    s = int(total_time % 60)
    m = int(total_time / 60) % 60
    h = int(total_time / 60 / 60) % 24
    d = int(total_time / 60 / 60 / 24)
    return '{:0>2d}d-{:0>2d}h-{:0>2d}m-{:0>2d}s'.format(d, h, m, s)


def get_current_timestamp():
    ct = time()
    ms = int((ct - int(ct)) * 1000)
    return '[ {},{:0>3d} ] '.format(strftime('%Y-%m-%d %H:%M:%S', localtime(ct)), ms)


def load_checkpoint(work_dir, model_name='resume', fold_idx=-1):
    if model_name == 'resume':
        file_name = '{}/checkpoint.pth.tar'.format(work_dir)
    elif model_name == 'debug':
        file_name = '{}/temp/debug.pth.tar'.format(work_dir)
    else:
        dirs, accs, labels = {}, {}, {}
        model_dir = '{}/{}'.format(work_dir, model_name)

        if fold_idx >= 0:
            # ====== K-Fold 模式：自動載入含有指定 fold_idx 的最新模型 ======
            if os.path.exists(model_dir):
                for dir_time in sorted(os.listdir(model_dir)):
                    full_path = os.path.join(model_dir, dir_time)
                    if os.path.isdir(full_path):
                        # 檢查 fold_results.json 是否匹配
                        fold_file = os.path.join(full_path, 'fold_results.json')
                        if os.path.exists(fold_file):
                            with open(fold_file, 'r') as f:
                                fold_info = json.load(f)
                            if fold_info.get('fold_idx') == fold_idx:
                                state_file = os.path.join(full_path, 'reco_results.json')
                                if os.path.exists(state_file):
                                    with open(state_file, 'r') as f:
                                        best_state = json.load(f)
                                    key = str(len(dirs) + 1)
                                    accs[key] = best_state.get('mae', best_state.get('acc_top1', 0))
                                    dirs[key] = dir_time
                                    labels[key] = '[fold {}] {}'.format(fold_idx, dir_time)

            if len(dirs) == 0:
                logging.warning('Warning: No model found for fold {}!'.format(fold_idx))
                return None

            # 自動選擇最新的模型（sorted by timestamp，取最後一個）
            latest_key = max(dirs.keys(), key=lambda k: int(k))
            logging.info('K-Fold mode: auto-loading latest model for fold {}'.format(fold_idx))
            metric_name = 'MAE' if 'mae' in best_state else 'accuracy'
            metric_format = '{:.4f}' if 'mae' in best_state else '{:.2%}'
            logging.info('  -> {}: {} | {}'.format(metric_name, metric_format.format(accs[latest_key]), labels[latest_key]))
            file_name = '{}/{}/{}.pth.tar'.format(model_dir, dirs[latest_key], model_name)

        else:
            # ====== 一般模式：列出所有模型供手動選擇 ======
            if os.path.exists(model_dir):
                for dir_time in sorted(os.listdir(model_dir)):
                    entry_path = os.path.join(model_dir, dir_time)
                    if os.path.isdir(entry_path):
                        state_file = os.path.join(entry_path, 'reco_results.json')
                        if os.path.exists(state_file):
                            with open(state_file, 'r') as f:
                                best_state = json.load(f)
                            key = str(len(dirs) + 1)
                            accs[key] = best_state.get('mae', best_state.get('acc_top1', 0))
                            dirs[key] = dir_time
                            # 如果有 fold_results.json，加上 fold 標籤
                            fold_file = os.path.join(entry_path, 'fold_results.json')
                            if os.path.exists(fold_file):
                                with open(fold_file, 'r') as f:
                                    fold_info = json.load(f)
                                labels[key] = '[fold {}] {}'.format(fold_info.get('fold_idx', '?'), dir_time)
                            else:
                                labels[key] = dir_time

            if len(dirs) == 0:
                logging.warning('Warning: Do NOT exists any model in workdir!')
                logging.info('Evaluating initial or pretrained model.')
                return None

            logging.info('Please choose the evaluating model from the following models.')
            logging.info('Default is the initial or pretrained model.')
            for key in dirs.keys():
                metric_name = 'MAE' if 'mae' in best_state else 'accuracy'
                metric_format = '{:.4f}' if 'mae' in best_state else '{:.2%}'
                logging.info('({}) {}: {} | {}'.format(key, metric_name, metric_format.format(accs[key]), labels[key]))
            logging.info('Your choice (number of the model, q for quit): ')
            while True:
                idx = input(get_current_timestamp())
                if idx == '':
                    logging.info('Evaluating initial or pretrained model.')
                    return None
                elif idx in dirs.keys():
                    break
                elif idx == 'q':
                    logging.info('Quit!')
                    sys.exit(1)
                else:
                    logging.info('Wrong choice!')
            file_name = '{}/{}/{}.pth.tar'.format(model_dir, dirs[idx], model_name)

    if os.path.exists(file_name):
        return torch.load(file_name, map_location=torch.device('cpu'), weights_only=False)
    else:
        logging.info('')
        logging.error('Error: Do NOT exist this checkpoint: {}!'.format(file_name))
        raise ValueError()


def save_checkpoint(model, optimizer, scheduler, epoch, best_state, is_best, work_dir, save_dir, model_name, save_interval=0):
    """
    Save checkpoint with optional periodic epoch saving.
    
    Args:
        save_interval: If > 0, save epoch checkpoint every N epochs (e.g., epoch_30.pth.tar)
    """
    for key in model.keys():
        model[key] = model[key].cpu()
    checkpoint = {
        'model': model, 'optimizer': optimizer, 'scheduler': scheduler,
        'best_state': best_state, 'epoch': epoch,
    }
    cp_name = '{}/checkpoint.pth.tar'.format(work_dir)
    torch.save(checkpoint, cp_name)
    
    # Save best model
    if is_best:
        shutil.copy(cp_name, '{}/{}.pth.tar'.format(save_dir, model_name))
        with open('{}/reco_results.json'.format(save_dir), 'w') as f:
            # 對於回歸任務，不刪除 'cm' 鍵，因為它不存在
            best_state_copy = best_state.copy()
            if 'cm' in best_state_copy:
                del best_state_copy['cm']
            json.dump(best_state_copy, f)
    
    # Save epoch checkpoint if save_interval is configured
    if save_interval > 0 and epoch % save_interval == 0:
        epoch_checkpoint_path = '{}/epoch_{}.pth.tar'.format(save_dir, epoch)
        shutil.copy(cp_name, epoch_checkpoint_path)
        logging.info('Saved epoch checkpoint: {}'.format(epoch_checkpoint_path))


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
