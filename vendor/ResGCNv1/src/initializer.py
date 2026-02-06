import os, warnings, logging, pynvml, torch, numpy as np
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from . import utils as U
from . import dataset
from . import model
from . import scheduler


class Initializer():
    def __init__(self, args, save_dir, inference_only=False):
        self.args = args
        self.save_dir = save_dir
        self.inference_only = inference_only

        logging.info('')
        logging.info('Starting preparing ...')
        self.init_environment()
        self.init_device()
        
        if self.inference_only:
            # Inference mode: only load metadata (skip heavy data loading)
            self.init_metadata()
            logging.info('Initializer: Inference mode (fast path)')
        else:
            # Training mode: full initialization
            self.init_dataloader()
            
        self.init_model()
        
        if not self.inference_only:
            self.init_optimizer()
            self.init_lr_scheduler()
            self.init_loss_func()
            
        logging.info('Successful!')
        logging.info('')

    def init_metadata(self):
        """Initialize dataset metadata without loading data (for inference)"""
        # Get metadata from dataset module
        self.data_shape, self.num_class, self.A, self.parts = dataset.get_dataset_info(self.args.dataset)
        
        # Determine num_class from config if provided (override default)
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args.get(dataset_name, {})
        if 'num_class' in dataset_args:
             self.num_class = dataset_args['num_class']

        # Setup bin_centers for LDL if needed
        custom_bin_centers = dataset_args.get('custom_bin_centers', None)
        if custom_bin_centers:
            self.bin_centers = torch.tensor(custom_bin_centers, dtype=torch.float32)
        else:
            self.bin_centers = None
            
        logging.info('Data shape: {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))

    def init_environment(self):
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.train_mae_values = []
        self.val_mae_values = []
        if self.args.debug:
            self.no_progress_bar = True
            self.model_name = 'debug'
            self.scalar_writer = None
        elif self.args.evaluate or self.args.extract or getattr(self.args, 'predict', False):
            self.no_progress_bar = self.args.no_progress_bar
            config_name = os.path.splitext(os.path.basename(self.args.config))[0]
            self.model_name = '{}_{}_{}'.format(config_name, self.args.model_type, self.args.dataset)
            self.scalar_writer = None
            warnings.filterwarnings('ignore')
        else:
            self.no_progress_bar = self.args.no_progress_bar
            config_name = os.path.splitext(os.path.basename(self.args.config))[0]
            self.model_name = '{}_{}_{}'.format(config_name, self.args.model_type, self.args.dataset)
            self.scalar_writer = SummaryWriter(logdir=self.save_dir)
            warnings.filterwarnings('ignore')
        logging.info('Saving model name: {}'.format(self.model_name))

    def init_device(self):
        if type(self.args.gpus) is int:
            self.args.gpus = [self.args.gpus]
        
        # Check if CUDA is actually available
        if not torch.cuda.is_available():
            if len(self.args.gpus) > 0:
                logging.warning(f'CUDA is not available but GPUs {self.args.gpus} were requested. Falling back to CPU.')
            else:
                logging.info('CUDA is not available. Using CPU.')
            
            self.args.gpus = [] # Clear GPUs to force CPU mode
        
        if len(self.args.gpus) > 0 and torch.cuda.is_available():
            # Try to use pynvml for GPU memory monitoring (optional, may fail on Windows)
            try:
                pynvml.nvmlInit()
                for i in self.args.gpus:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memused = meminfo.used / 1024 / 1024
                    logging.info('GPU-{} used: {}MB'.format(i, memused))
                    if memused > 4000:  # 增加記憶體限制到4GB
                        pynvml.nvmlShutdown()
                        logging.info('')
                        logging.error('GPU-{} is occupied!'.format(i))
                        raise ValueError()
                pynvml.nvmlShutdown()
            except Exception as e:
                logging.warning('NVML not available, skipping GPU memory check: {}'.format(type(e).__name__))
            self.output_device = self.args.gpus[0]
            self.device =  torch.device('cuda:{}'.format(self.output_device))
            torch.cuda.set_device(self.output_device)
        else:
            logging.info('Using CPU!')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            self.output_device = None
            self.device =  torch.device('cpu')

    def init_dataloader(self):
        dataset_name = self.args.dataset.split('-')[0]
        dataset_args = self.args.dataset_args[dataset_name]
        self.train_batch_size = dataset_args['train_batch_size']
        self.eval_batch_size = dataset_args['eval_batch_size']
        self.feeders, self.data_shape, self.num_class, self.A, self.parts = dataset.create(
            self.args.debug, self.args.dataset, **dataset_args
        )
        # Get bin_centers for LDL evaluation
        bin_centers_np = getattr(self.feeders['train'], 'bin_centers', None)
        self.bin_centers = torch.tensor(bin_centers_np, dtype=torch.float32) if bin_centers_np is not None else None
        self.train_loader = DataLoader(self.feeders['train'],
            batch_size=self.train_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=True, drop_last=True
        )
        self.eval_loader = DataLoader(self.feeders['eval'],
            batch_size=self.eval_batch_size, num_workers=4*len(self.args.gpus),
            pin_memory=True, shuffle=False, drop_last=False
        )
        self.location_loader = self.feeders['ntu_location'] if dataset_name == 'ntu' else None
        logging.info('Dataset: {}'.format(self.args.dataset))
        logging.info('Batch size: train-{}, eval-{}'.format(self.train_batch_size, self.eval_batch_size))
        logging.info('Data shape (branch, channel, frame, joint, person): {}'.format(self.data_shape))
        logging.info('Number of action classes: {}'.format(self.num_class))

    def init_model(self):
        kwargs = {
            'data_shape': self.data_shape,
            'num_class': self.num_class,
            'A': torch.Tensor(self.A),
            'parts': [torch.Tensor(part).long() for part in self.parts]
        }
        self.model = model.create(self.args.model_type, **(self.args.model_args), **kwargs).to(self.device)
        
        if self.device.type == 'cpu':
            class CPUWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.module = model
                def forward(self, *args, **kwargs):
                    return self.module(*args, **kwargs)
            
            self.model = CPUWrapper(self.model)
            logging.info('Wrapped model with CPUWrapper for compatibility')
        else:
            self.model = torch.nn.DataParallel(
                self.model, device_ids=self.args.gpus, output_device=self.output_device
            )
            
        logging.info('Model: {} {}'.format(self.args.model_type, self.args.model_args))
        logging.info('Model parameters: {:.2f}M'.format(
            sum(p.numel() for p in self.model.parameters()) / 1000 / 1000
        ))
        # Load pretrained weights
        if os.path.isfile(self.args.pretrained_path):
             pretrained_model = self.args.pretrained_path
        else:
             pretrained_model = '{}/{}.pth.tar'.format(self.args.pretrained_path, self.model_name)

        if os.path.exists(pretrained_model):
            # weights_only=False needed for PyTorch 2.6+ (checkpoint contains numpy objects)
            checkpoint = torch.load(pretrained_model, map_location=torch.device('cpu'), weights_only=False)
            
            # Handle DataParallel prefix match if needed (Initializer uses DataParallel so typically matches)
            # But just in case weights don't have 'module.' prefix or vice versa, standard loading might need care
            # Usually checkpoint['model'] matches the architecture if trained with same code
            
            self.model.module.load_state_dict(checkpoint['model'])
            logging.info('Pretrained model: {}'.format(pretrained_model))
        elif self.args.pretrained_path:
            logging.warning('Warning: Do NOT exist this pretrained model: {}'.format(pretrained_model))

    def init_optimizer(self):
        try:
            optimizer = U.import_class('torch.optim.{}'.format(self.args.optimizer))
        except:
            logging.info('Do NOT exist this optimizer: {}!'.format(self.args.optimizer))
            logging.info('Try to use SGD optimizer.')
            self.args.optimizer = 'SGD'
            optimizer = U.import_class('torch.optim.SGD')
        optimizer_args = self.args.optimizer_args[self.args.optimizer]
        self.optimizer = optimizer([
            {'params': [p for n, p in self.model.named_parameters() if p.dim() >= 2], 'weight_decay': optimizer_args.get('weight_decay', 0.0)},
            {'params': [p for n, p in self.model.named_parameters() if p.dim() < 2], 'weight_decay': 0.0}
        ], **{k: v for k, v in optimizer_args.items() if k != 'weight_decay'})
        logging.info('Optimizer: {} {}'.format(self.args.optimizer, optimizer_args))

    def init_lr_scheduler(self):
        scheduler_args = self.args.scheduler_args[self.args.lr_scheduler]
        self.max_epoch = scheduler_args['max_epoch']
        base_lr = self.optimizer.param_groups[0]['lr']
        scheduler_args['base_lr'] = base_lr
        lr_scheduler = scheduler.create(self.args.lr_scheduler, len(self.train_loader), **scheduler_args)
        self.eval_interval, lr_lambda = lr_scheduler.get_lambda()
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        logging.info('LR_Scheduler: {} {}'.format(self.args.lr_scheduler, scheduler_args))

    def init_loss_func(self):
        # Check if LDL is enabled in config
        use_ldl = getattr(self.args, 'use_ldl', False)
        loss_type = getattr(self.args, 'loss_type', 'ce')
        reg_weight = getattr(self.args, 'reg_weight', 0.0)

        if use_ldl:
            if reg_weight > 0:
                from .losses import LDLLossWithL1
                self.loss_func = LDLLossWithL1(reg_weight, self.bin_centers.to(self.device)).to(self.device)
                logging.info('Loss function: LDLLossWithL1 (LDL enabled, reg_weight={})'.format(reg_weight))
            elif loss_type.lower() == 'kl':
                from .losses import KLDivergenceLoss
                self.loss_func = KLDivergenceLoss().to(self.device)
                logging.info('Loss function: KLDivergenceLoss (LDL enabled)')
            elif loss_type.lower() == 'ordinal' or loss_type.lower() == 'ordinal_kl_emd':
                # Combined KL + EMD loss (排序優先)
                from .losses import OrdinalKLEMDLoss
                ordinal_weight = getattr(self.args, 'ordinal_weight', 1.0)
                self.loss_func = OrdinalKLEMDLoss(ordinal_weight=ordinal_weight).to(self.device)
                logging.info('Loss function: OrdinalKLEMDLoss (KL + {}*EMD)'.format(ordinal_weight))
            elif loss_type.lower() == 'ordinal_emd':
                # Pure EMD loss
                from .losses import OrdinalEMDLoss
                self.loss_func = OrdinalEMDLoss(emd_loss_type='mse').to(self.device)
                logging.info('Loss function: OrdinalEMDLoss (pure EMD)')
            else:
                from .losses import ExpectationLoss
                self.loss_func = ExpectationLoss(base_loss=loss_type).to(self.device)
                logging.info('Loss function: ExpectationLoss with {} base loss (LDL enabled)'.format(loss_type.upper()))
        else:
            self.loss_func = torch.nn.CrossEntropyLoss().to(self.device)
            logging.info('Loss function: {}'.format(self.loss_func.__class__.__name__))
