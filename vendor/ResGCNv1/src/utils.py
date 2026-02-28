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


def load_checkpoint(work_dir, model_name='resume'):
    if model_name == 'resume':
        file_name = '{}/checkpoint.pth.tar'.format(work_dir)
    elif model_name == 'debug':
        file_name = '{}/temp/debug.pth.tar'.format(work_dir)
    else:
        dirs, accs = {}, {}
        work_dir = '{}/{}'.format(work_dir, model_name)
        if os.path.exists(work_dir):
            for i, dir_time in enumerate(os.listdir(work_dir)):
                if os.path.isdir('{}/{}'.format(work_dir, dir_time)):
                    state_file = '{}/{}/reco_results.json'.format(work_dir, dir_time)
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            best_state = json.load(f)
                        # 對於回歸任務，使用 'mae'；對於分類任務，使用 'acc_top1'
                        accs[str(i+1)] = best_state.get('mae', best_state.get('acc_top1', 0))
                        dirs[str(i+1)] = dir_time
        if len(dirs) == 0:
            logging.warning('Warning: Do NOT exists any model in workdir!')
            logging.info('Evaluating initial or pretrained model.')
            return None
        logging.info('Please choose the evaluating model from the following models.')
        logging.info('Default is the initial or pretrained model.')
        for key in dirs.keys():
            # 對於回歸任務，顯示 MAE；對於分類任務，顯示準確率
            metric_name = 'MAE' if 'mae' in best_state else 'accuracy'
            metric_format = '{:.4f}' if 'mae' in best_state else '{:.2%}'
            logging.info('({}) {}: {} | training time: {}'.format(key, metric_name, metric_format.format(accs[key]), dirs[key]))
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
        file_name = '{}/{}/{}.pth.tar'.format(work_dir, dirs[idx], model_name)
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
