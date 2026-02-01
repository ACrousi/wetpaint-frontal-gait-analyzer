import os, yaml, argparse
from time import strftime

from src import utils as U
from src.generator import Generator
from src.processor import Processor
from src.visualizer import Visualizer


def main():
    # Loading Parameters
    parser = init_parameters()
    args, _ = parser.parse_known_args()

    # Updating Parameters (cmd > yaml > default)
    args = update_parameters(parser, args)

    # Setting save_dir
    save_dir = get_save_dir(args)
    U.set_logging(save_dir)
    with open('{}/config.yaml'.format(save_dir), 'w') as f:
        yaml.dump(vars(args), f)

    # Processing
    if args.generate_data or args.generate_label:
        g = Generator(args)
        g.start()

    elif args.predict:
        # Predict mode: inference on input JSON files
        # 改為直接使用 inference 模組，避免 Processor 初始化帶來的額外開銷 (如載入訓練數據)
        from src.inference import ResGCNInference
        import json
        
        # 直接從 args 建立 inference pipeline
        inference = ResGCNInference.from_config(args)
        
        # 執行預測
        predictions = inference.predict_batch(args.input_json)
        
        # 轉換結果並輸出 JSON (供 subprocess 讀取)
        results = [pred.to_dict() for pred in predictions]
        print('===PREDICTION_RESULTS_START===')
        print(json.dumps(results, ensure_ascii=False, indent=2))
        print('===PREDICTION_RESULTS_END===')

    elif args.extract or args.visualization:
        if args.extract:
            p = Processor(args, save_dir)
            p.extract()
        if args.visualization:
            v = Visualizer(args)
            v.start()

    else:
        p = Processor(args, save_dir)
        p.start()


def init_parameters():
    parser = argparse.ArgumentParser(description='Skeleton-based Action Recognition')

    # Setting
    parser.add_argument('--config', '-c', type=str, default='', help='ID of the using config', required=True)
    parser.add_argument('--gpus', '-g', type=int, nargs='+', default=[], help='Using GPUs')
    parser.add_argument('--seed', '-s', type=int, default=1, help='Random seed')
    parser.add_argument('--debug', '-db', default=False, action='store_true', help='Debug')
    parser.add_argument('--pretrained_path', '-pp', type=str, default='', help='Path to pretrained models')
    parser.add_argument('--work_dir', '-w', type=str, default='', help='Work dir')
    parser.add_argument('--no_progress_bar', '-np', default=False, action='store_true', help='Do not show progress bar')
    parser.add_argument('--path', '-p', type=str, default='', help='Path to save preprocessed skeleton files')

    # Processing
    parser.add_argument('--resume', '-r', default=False, action='store_true', help='Resume from checkpoint')
    parser.add_argument('--evaluate', '-e', default=False, action='store_true', help='Evaluate')
    parser.add_argument('--extract', '-ex', default=False, action='store_true', help='Extract')
    parser.add_argument('--visualization', '-v', default=False, action='store_true', help='Visualization')
    parser.add_argument('--generate_data', '-gd', default=False, action='store_true', help='Generate skeleton data')
    parser.add_argument('--generate_label', '-gl', default=False, action='store_true', help='Only generate label')
    parser.add_argument('--predict', '-pr', default=False, action='store_true', help='Predict mode for inference')
    parser.add_argument('--input_json', '-ij', type=str, nargs='+', default=[], help='Input JSON files for prediction')

    # Visualization
    parser.add_argument('--visualization_class', '-vc', type=int, default=0, help='Class: 1 ~ 60, 0 means true class')
    parser.add_argument('--visualization_sample', '-vs', type=int, default=0, help='Sample: 0 ~ batch_size-1')
    parser.add_argument('--visualization_frames', '-vf', type=int, nargs='+', default=[], help='Frame: 0 ~ max_frame-1')
    parser.add_argument('--visualization_heatmap_save', '-vhs', type=str, default='./heatmaps_output', help='Path to save all validation heatmaps')

    # Dataloader
    parser.add_argument('--dataset', '-d', type=str, default='', help='Select dataset')
    parser.add_argument('--dataset_args', default=dict(), help='Args for creating dataset')

    # Model
    parser.add_argument('--model_type', '-mt', type=str, default='', help='Model type')
    parser.add_argument('--model_args', default=dict(), help='Args for creating model')

    # Optimizer
    parser.add_argument('--optimizer', '-o', type=str, default='', help='Initial optimizer')
    parser.add_argument('--optimizer_args', default=dict(), help='Args for optimizer')

    # LR_Scheduler
    parser.add_argument('--lr_scheduler', '-ls', type=str, default='', help='Initial learning rate scheduler')
    parser.add_argument('--scheduler_args', default=dict(), help='Args for scheduler')

    # LDL (Label Distribution Learning)
    parser.add_argument('--use_ldl', default=False, action='store_true', help='Use Label Distribution Learning')
    parser.add_argument('--loss_type', type=str, default='ce', help='Loss type for LDL: mse, mae, or kl')
    parser.add_argument('--ldl_sigma', type=float, default=1.0, help='Sigma parameter for LDL distributions')
    parser.add_argument('--reg_weight', type=float, default=0.0, help='Regularization weight for LDL (e.g. MAE)')
    parser.add_argument('--ordinal_weight', type=float, default=1.0, help='Weight for ordinal (EMD) term in OrdinalLoss')
    parser.add_argument('--early_stop_patience', type=int, default=100000, help='Early stopping patience')
    parser.add_argument('--save_interval', type=int, default=0, help='Save model checkpoint every N epochs (0 = disabled)')
    parser.add_argument('--best_metric_criterion', type=str, default='mae', help='Best model criterion for LDL: mae or spearman')

    return parser


def update_parameters(parser, args):
    # 優先使用給定的完整路徑，若不存在才回退到 ./configs/{}.yaml
    if os.path.exists(args.config):
        config_path = args.config
    else:
        config_path = './configs/{}.yaml'.format(args.config)
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            
            # 提取 workspace_root（非命令行參數，單獨處理）
            workspace_root_from_yaml = yaml_arg.pop('workspace_root', None)
            
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Config file not found: {} (also tried ./configs/{}.yaml)'.format(args.config, args.config))
    
    args = parser.parse_args()
    
    # ====================================
    # Workspace 路徑解析
    # ====================================
    # 優先順序: 環境變數 > yaml config > 預設值
    workspace_env = os.environ.get('WETPAINT_WORKSPACE')
    if workspace_env:
        workspace_root = os.path.abspath(workspace_env)
    else:
        # 從 yaml 取得 workspace_root，預設為 "../../outputs"（相對於 configs 目錄）
        ws_root = workspace_root_from_yaml if workspace_root_from_yaml else '../../outputs'
        # 以 config 文件位置為基準解析相對路徑
        config_dir = os.path.dirname(os.path.abspath(config_path))
        workspace_root = os.path.abspath(os.path.join(config_dir, ws_root))
    
    # 確保 workspace 存在
    os.makedirs(workspace_root, exist_ok=True)
    
    # 解析 work_dir（輸出目錄）
    if args.work_dir and not os.path.isabs(args.work_dir):
        args.work_dir = os.path.join(workspace_root, args.work_dir)
    os.makedirs(args.work_dir, exist_ok=True)
    
    # 解析 dataset_args 中的路徑
    if hasattr(args, 'dataset_args') and isinstance(args.dataset_args, dict):
        for dataset_name, dataset_config in args.dataset_args.items():
            if isinstance(dataset_config, dict):
                path_keys = ['input_path', 'path', 'metadata_path', 
                            'data_path', 'label_path', 
                            'eval_data_path', 'eval_label_path']
                for key in path_keys:
                    if key in dataset_config:
                        val = dataset_config[key]
                        # 空字串表示 workspace_root 本身
                        if val == '':
                            dataset_config[key] = workspace_root
                        elif val and not os.path.isabs(val):
                            dataset_config[key] = os.path.join(workspace_root, val)
    
    return args


def get_save_dir(args):
    if args.debug or args.evaluate or args.extract or args.visualization or args.generate_data or args.generate_label or getattr(args, 'predict', False):
        save_dir = '{}/temp'.format(args.work_dir)
    else:
        ct = strftime('%Y-%m-%d %H-%M-%S')
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        save_dir = '{}/{}_{}_{}/{}'.format(args.work_dir, config_name, args.model_type, args.dataset, ct)
    U.create_folder(save_dir)
    return save_dir


if __name__ == '__main__':
    os.chdir(os.getcwd())
    main()
