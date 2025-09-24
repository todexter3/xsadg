import argparse
import os
import torch
from exp.exp_multiple_regression_kfold import Exp_Multiple_Regression_Fold
import random
import numpy as np
import os
from utils.str2bool import str2bool

os.environ["KMP_AFFINITY"] = "noverbose"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

torch.set_num_threads(8)

def main():
    if args.is_training:
        # flag = 1
        for args.tau_hat_init in [4.5, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -4.5]: # [-5,5]之间
            for args.learning_rate in [1e-5,1e-4,1e-3]:
                for args.batch_size in [256]:
                    for args.seq_len in [120]:
                        # ModernTCN
                        # for args.decomposition in [0, 1]:
                        #                 args.large_size = [51]
                        #                 args.small_size = [5]
                        #                 args.num_blocks = [1]
                        #                 args.patch_size = 8
                        #                 args.ffn_ratio = 8
                        #                 args.stride = 4
                                        # args.ps = f'phiy{args.tau_hat_init}_{args.decomposition}'
                        # fits
                        # for args.H_order in [3]:
                        #                 args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10
                        #                 args.ps = f'phi1{args.tau_hat_init}_{args.H_order}'
                        # --mlp LSTM
                        # for args.MLP_layers in [3]:
                        #     for args.MLP_hidden in [128]:
                        #                 args.ps = f'phi1{args.tau_hat_init}'
                            # --patchtst duet itrans
                                for args.d_model in [128]: 
                                    args.d_ff = int(args.d_model * 2)
                                    for args.patch_len in [16]:
                                        args.stride = int(args.patch_len / 2)
                                    # for args.e_layers in [3]:
                                    #     args.e_layers = 3
                                        args.ps = f'phiyhis{args.tau_hat_init}'
                        # for args.d_model in [64, 128]: # wpmixer
                        #     for args.d_ff in [16]:
                        #         for args.patch_len in [16]:
                        #             args.stride = 8
                        #             for args.wavelet in ['db2', 'sym4', 'coif4']:
                                        # args.val = f'phi{args.tau_hat_init}_{args.wavelet}'
                            # for args.d_model in [8, 4]: # path
                            #     for args.d_ff in [32, 16]:
                            #             args.layer_nums = 2
                                        if args.data_type == 'daily':
                                            args.pred_task = 10 # y1
                                            args.pred_len = 1

                                        set_seed(args.seed)
                                        args.size = [args.seq_len, args.pred_len]
                                        
                                        train_des = f"task{args.task_name}_{args.model}_test_year{args.test_year}_mlp{args.MLP_hidden}_lay{args.MLP_layers}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_freq{args.freq}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                                        model_des = f"dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}"
                                        patching_des = f'_pl{args.patch_len}_sr{args.stride}_ps{args.ps}'
                                        setting = train_des + model_des + patching_des
                                        
                                        args.save_path = os.path.join(save_path, f'y{args.pred_task}/{args.model}_{setting}')
                                        args.checkpoints = args.save_path
                                        args.logs_dir = args.save_path + f'/logs'
                                        if os.path.exists(args.save_path + '/pred.npy'):
                                            continue
                                        if not os.path.exists(args.save_path):
                                            os.makedirs(args.save_path)
                                        print('Args in experiment:')
                                        print(args)
                                        with open(f'{args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                                            file.write('Args in experiment:\n' + f'{args}\n\n')
                                        
                                        Exp = Exp_Multiple_Regression_Fold
                                        exp = Exp(args)  # set experiments
                                        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                                        exp.train(setting)
                                        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                                        exp.test(setting)
                                        torch.cuda.empty_cache()
    else:
        for args.tau_hat_init in [0.0]: # [-5,5]之间
            # args.ps = f'phi{args.tau_hat_init}'
            for args.learning_rate in [1e-3]:
            # for args.learning_rate in [1e-4]:
                for args.batch_size in [128]:
                    for args.seq_len in [120]:
                    # for args.seq_len in range(180, 210+30,30): # min15 720
                        # --mlp LSTM
                        for args.MLP_layers in [3]:
                            for args.MLP_hidden in [128]:
                                        args.ps = f'phi1{args.tau_hat_init}'
                        # for args.MLP_layers in [5,6,7]:
                        #     for args.MLP_hidden in [256,512,1024]:
                        # for args.MLP_layers in [4]:
                        #     for args.MLP_hidden in [64]:
                            # --patchtst duet
                                # for args.d_model in [128]: 
                                #     args.d_ff = int(args.d_model * 2)
                                    # for args.patch_len in [16,32]:
                                #         args.stride = int(args.patch_len / 2)
                                    # for args.e_layers in [3]:
                                        # args.e_layers = 3
                                        # args.ps = f'phi{args.tau_hat_init}_{args.e_layers}'
                                        # if flag == 1:
                                        #     flag = 2
                                        #     continue
                        # for args.d_model in [64, 128]: # wpmixer
                        #     for args.d_ff in [16]:
                        #         for args.patch_len in [16]:
                        #             args.stride = 8
                        #             for args.wavelet in ['db2', 'sym4', 'coif4']:
                                        # args.val = f'phi{args.tau_hat_init}_{args.wavelet}'
                            # for args.d_model in [4, 8]: # path
                            #     for args.d_ff in [16, 32]:
                            #             args.layer_nums = 2
                                        print('Args in experiment:')
                                        print(args)
                                        if args.data_type == 'daily':
                                            args.pred_task = 10
                                            args.pred_len = 1
                                        set_seed(args.seed)
                                        args.size = [args.seq_len, args.pred_len]
                                        # model = Model(args)
                                        train_des = f"task{args.task_name}_{args.model}_test_year{args.test_year}_mlp{args.MLP_hidden}_lay{args.MLP_layers}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_freq{args.freq}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                                        model_des = f"dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}"
                                        patching_des = f'_pl{args.patch_len}_sr{args.stride}_ps{args.ps}'
                                        setting = train_des + model_des + patching_des
                                        args.save_path = os.path.join(save_path, f'y{args.pred_task}/{args.model}_{setting}')
                                        args.checkpoints = args.save_path
                                        args.logs_dir = args.save_path + f'/logs'
                                        if not os.path.exists(args.save_path):
                                            os.makedirs(args.save_path)
                                        with open(f'{args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                                            file.write('Args in experiment:\n' + f'{args}\n\n')
                                        
                                        Exp = Exp_Multiple_Regression_Fold
                                        exp = Exp(args)  # set experiments
                                        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                                        exp.test(setting)
                                        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--task_name', type=str, default='multiple_regression',
                        help='task name, options:[Long_term_forecasting, anomaly_detection, predict_feature,multiple_regression, LGB]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='FC_MLP',
                        help='model name, options: [GPT2TS, ]')
                        
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    # data loader
    parser.add_argument('--dataset', type=str, default='ETTh1',
                        help='[ETTh1, ETTh2, ETTm1, ETTm2, weather, psm, smap]')
    parser.add_argument('--prompt',type=str, default='Etth1')
    parser.add_argument('--root_path', type=str, default='/home/liangxijie1/phi-2/dataset/',
                        help='root path of the data file:feature_1419_5, d1')
    parser.add_argument('--data_path', type=str, default='LongtermForecast/ETT-small/',
                        help='data file, options: [ETT-small, electricity, exchange_rate, illness, traffic, weather]')
    parser.add_argument('--freq', type=str, default='t',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints_heiyi/', help='location of model checkpoints')

    parser.add_argument('--drop_ratio', type=float, default=0.2, help='Set a dropping ratio for feature_selection')
    parser.add_argument('--train_data_start_year', type=int, default=2010)
    parser.add_argument('--test_data_start_year', type=int, default=2021)
    parser.add_argument('--feature_selection',type = bool, default=False, help='whether to use feature selection')
    parser.add_argument('--extra_input',type = bool, default=False, help='whether to add tikcter')

    # Forecast task
    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    parser.add_argument('--patch_len', type=int, default=8)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=16)

    #ModernTCN
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
    # parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
    # parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')

    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+',type=int, default=[64,64,64,64], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256])

    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
    parser.add_argument('--use_multi_scale', type=str2bool, default=False, help='use_multi_scale fusion')

    # MLP
    parser.add_argument('--MLP_hidden', type=int, default=32,
                        help='The middle tier scale of fc MLPn in ecoder')
    parser.add_argument('--MLP_layers', type=int, default=2, help='layers of MLP')
    parser.add_argument('--max_depth', type=int, default=2, help='kernel size of fc conv')
    parser.add_argument('--weight_std', type=float, default=0.01, help='weight initializes standard deviation')

    # timemixer++
    parser.add_argument('--down_sampling_layers', type=int, default=2, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--channel_mixing', type=int, default=1,
                        help='0: channel mixing 1: whether to use channel_mixing')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # DUET
    parser.add_argument('--noisy_gating', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--num_experts', type=int, default=4)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--hidden_size', type=int, default=256)

    # PDF
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--kernel_size', type=int, default=5, help='kernel size of fc conv')
    parser.add_argument('--embed_type', type=int, default=0,
                        help='0: default 1: value patch_embedding + temporal patch_embedding + positional patch_embedding 2: value '
                            'patch_embedding + temporal patch_embedding 3: value patch_embedding + positional patch_embedding 4: value patch_embedding')
    parser.add_argument('--period', type=int, nargs='+', default=[24], help='period list')
    parser.add_argument('--individual', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--fc_dropout', type=float, default=0.0, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--add', action='store_true', default=False, help='add')
    parser.add_argument('--wo_conv', action='store_true', default=False, help='without convolution')
    parser.add_argument('--serial_conv', action='store_true', default=False, help='serial convolution')
    parser.add_argument('--kernel_list', type=int, nargs='+', default=[3, 7, 9], help='kernel size list')

    # pathformer
    parser.add_argument('--batch_norm', type=int, default=0)
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--layer_nums', type=int, default=3)
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16,12,8,32,12,8,6,4,8,6,4,2])
    parser.add_argument('--num_nodes', type=int, default=9)

    parser.add_argument('--tfactor', type = int, default = 5, help = 'expansion factor in the patch mixer')
    parser.add_argument('--dfactor', type = int, default = 5, help = 'expansion factor in the embedding mixer')
    parser.add_argument('--wavelet', type = str, default = 'db2', help = 'wavelet type for wavelet transform')
    parser.add_argument('--level', type = int, default = 1, help = 'level for multi-level wavelet decomposition')
    parser.add_argument('--use_amp', action = 'store_true', help = 'use automatic mixed precision training', default = False)
    parser.add_argument('--no_decomposition', action = 'store_true', default = False, help = 'whether to use wavelet decomposition')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--early_open', type=bool, default=True)
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='optimizer learning rate')
    parser.add_argument('--optim_type', type=str, default='Adam', help='select optimizer type, optional[SGD, Adam]')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay value')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function, optional[ MSE, MAE, CCC]')
    parser.add_argument('--lradj', type=str, default='type1',
                        help='adjust learning rate, optional:[type1, type2, not, cos, steplr]')
    parser.add_argument('--clip_value', type=float, default=0.5, help='clip grad')
    parser.add_argument('--pct_start', type=int, default=0.6)
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=4, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--dataset_num', type=str, default='0', help='AIOps have 29 dataset,number:0-28')

    # heiyi
    parser.add_argument('--save_path', type=str, default='/data/lrlresults/multiscale_patch', help='train start year')
    parser.add_argument('--train_start_year', type=str, default='2010', help='train start year')
    parser.add_argument('--train_end_year', type=str, default='2019', help='train end year')
    parser.add_argument('--val_start_year', type=str, default='2014', help='vali start year')
    parser.add_argument('--use_original_feature', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--kfold', action='store_true', help='use kfold', default=False)
    parser.add_argument('--per20', action='store_true', help='use foldper20', default=False)
    parser.add_argument('--num_fold', type=int, default=5, help='')
    parser.add_argument('--pred_task', type=int, default=10, help='y5,y10,y20')
    parser.add_argument('--lgb', action='store_true', help='use lgb regressor', default=False)
    parser.add_argument('--output_channels', type=int,default=1)

    parser.add_argument('--seed', type=int, default=42, help='seed')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    # args.use_multi_gpu=1
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist() # pathformer

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # args.is_training = 0
    args.data_new = '5'
    args.ticker_type = 2  #0,1,2(all)

    args.weight_decay = 1e-5
    args.drop_ratio = 0.1
    args.pct_start = 0.6

    args.feature_selection = False
    args.train_epochs = 60
    args.patience = 10
    # args.individual = True
    args.n_splits = 3
    args.dataset = 'heiyi'
    args.lradj = 'not'
    args.random_zero_prob = 0.0
    args.random_mask_prob = 0.0

    args.data_path = 'data/daily_label10_all_data_202312.h5' # daily 因子集和原始特征
    args.data_type = 'daily'
    args.freq = 'd'

    args.grad_norm = False
    args.dropout = args.drop_ratio

    args.train_start_year = '2010'
    # args.train_end_year = '2020'
    # args.gpu = 2
    args.test_year = str(int(args.train_end_year)+1)

    args.features = 'M' # long MS
    args.task_name = 'multiple_regression'  # [Long_term_forecasting, multiple_regression, predict_feature, classification]
    args.model = 'Path'  # wpmixer, PatchMLP, PatchTST, FC_MLP, Path, FC_Conv, FITS,MTPatchTST,MTMLP,LSTM, iTransformer
    args.kfold = True
    args.ps = True
    args.enc_in = 9
    args.num_fold = 5
    args.loss = 'MSE_with_weak'
    save_path = f'data/lrlresults/{args.data_type}/ori/kf/ptphiyhis256'

    main()