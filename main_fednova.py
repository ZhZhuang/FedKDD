from utils import get_logger
from core_fednova import fednova_train
from common import load_models,  init


#
# def load_models( args, config, logger ):
#     n_clients= config.n_clients
#
#     # 准备数据
#     path = config.dataset_path
#     if config.noniid:
#         # non iid   Dirichlet分布
#         if config.dataset == "Cifar10":
#             if config.dirichlet:
#                 # dirichlet 分布
#                 train_data_dict_list, test_data_dict_list = cifar_noniid_dirichlet(n_clients, alpha= config.alpha)
#             else:
#                 train_data_dict_list, test_data_dict_list = cifar_noniid(n_clients, path=path)
#         elif config.dataset == "FashionMNIST":
#             if config.dirichlet:
#                 pass
#                 # train_data_dict_list, _ = fmnist_noniid_dirichlet(n_clients, alpha=config.alpha)
#             else:
#                 train_data_dict_list, _ = fmnist_noniid(num_users= n_clients, path= path)
#     else:
#         # iid
#         if config.dataset == "Cifar10":
#             train_data_dict_list = get_user_data(n_clients, train=True, dataname="Cifar10")
#         elif config.dataset == "FashionMNIST":
#             train_data_dict_list = get_user_data(n_clients, train=True, dataname="FashionMNIST")
#
#
#     # 初始化用户
#     client_list = []
#
#     for i in range(n_clients):
#         in_ch = 3
#         if config.dataset != "Cifar10":
#             in_ch = 1
#         C_model = ResNet(ResidualBlock, [2, 2, 2, 2], 10, in_ch=in_ch)   # ResNet 18
#         data = train_data_dict_list[i]["sub_data"]
#         targets = train_data_dict_list[i]["sub_targets"]
#         test_data = test_data_dict_list[i]["sub_data"]
#         test_targets = test_data_dict_list[i]["sub_targets"]
#
#         ################# 获取每类样本的数量，并保存在文件中
#         count = [0 for _ in range(config.classes)]
#         for c in targets:  # lb_targets 为 0 ～ 9 ， 有操作
#             count[c] += 1
#         out = {"distribution": count }
#         output_file = f"{config.save_dir}/client_data_statistics_{i}.json"
#         # if not os.path.exists(output_file):
#         #     os.makedirs(output_file, exist_ok=True)
#         with open(output_file, 'w') as w:
#             json.dump(out, w)
#         #################
#
#         t_transform = transforms.Compose( [transforms.Resize(32),
#                                            transforms.ToTensor(),
#                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#                                            ] )
#         if config.dataset == "Cifar10":
#             transform = transforms.Compose([
#                                             transforms.Pad(4),
#                                             transforms.RandomHorizontalFlip(),  # ? 水平翻转
#                                             transforms.RandomCrop(32),
#                                             transforms.ToTensor(),
#                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#                                             ]
#                                            )
#         elif config.dataset == "FashionMNIST":
#             transform = transforms.Compose([
#                 transforms.Pad(4),
#                 transforms.RandomHorizontalFlip(),  # ? 水平翻转
#                 transforms.RandomCrop(28),
#                 transforms.ToTensor(),
#                 # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # error 变为 3 通道
#                 transforms.Normalize( 0.5, 0.5)
#             ]
#             )
#         dataset = BasicDataset(data, targets, transform=transform,onehot= False)
#         test_dataset = BasicDataset(test_data, test_targets, transform=t_transform,onehot= False)
#         dataloader = torch.utils.data.DataLoader(dataset, config.batch_size,
#                                                  shuffle=True,
#                                                  num_workers= 4)
#         test_dataloader = torch.utils.data.DataLoader(test_dataset, 256,
#                                                  shuffle=False,
#                                                  num_workers= 4)
#         client = Client(config,
#                         C_model=C_model,
#                         client_idx=i,
#                         dataloader= dataloader,
#                         dataset = dataset,
#                         t_dataloader= test_dataloader,
#                         logger= logger
#                         )
#
#         # Optimizers
#         # optimizer_C = torch.optim.SGD(C_model.parameters(),lr=0.001) 0.01
#         optimizer_C = torch.optim.Adam(C_model.parameters(), lr = config.lr)  #更适合resnet18
#         client.set_optimizer( optimizer_C )
#         client_list.append( client )
#
#     return client_list , transform
#
# def init():
#
#     # 添加参数
#     parser = argparse.ArgumentParser(description=globals()["__doc__"])  # ？
#     parser.add_argument(
#         "--config", type=str, default="fednova.yaml", help="Path to the config file"
#     )
#     parser.add_argument("--seed", type=int, default=1234, help="Random seed")
#     args = parser.parse_args()
#
#     # 读取配置文件
#     with open(os.path.join("configs", args.config), "r") as f:
#         config = yaml.safe_load(f)
#     new_config = dict2namespace(config)     #
#
#     # 保存路径
#     save_dir = None
#     if new_config.noniid == False:
#         save_dir = os.path.join(new_config.save_dir,
#                                 new_config.name + f"_{new_config.n_clients}" + \
#                                 f"_iid" + f"_{new_config.dataset}")
#     elif new_config.noniid and new_config.dirichlet == False:
#         save_dir = os.path.join(new_config.save_dir,
#                                 new_config.name + f"_{new_config.n_clients}" + \
#                                 f"_noniid" + f"_{new_config.dataset}")
#     elif new_config.noniid and new_config.dirichlet:
#         save_dir = os.path.join(new_config.save_dir,
#                                 new_config.name + f"_{new_config.n_clients}" + \
#                                 f"_dirichlet_alpha_{new_config.alpha}")
#     new_config.save_dir = save_dir
#
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir, exist_ok=True)    # True：创建目录的时候，如果已存在就不报错。
#
#     # 获取设备信息
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     logging.info("Using device: {}".format(device))
#     new_config.device = device
#
#     # 将配置信息保存
#     with open(os.path.join(new_config.save_dir, "config.yml"), "w") as f:
#         yaml.dump(new_config, f, default_flow_style=False)
#
#     seed = args.seed
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
#     torch.cuda.manual_seed_all(seed)
#
#     cudnn.deterministic = True  # 随机数种子seed确定时，模型的训练结果将始终保持一致
#     cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
#
#     return args , new_config
#
# def dict2namespace( config ):
#     namespace = argparse.Namespace()    # ？
#     for key, value in config.items():
#         if isinstance(value, dict):
#             new_value = dict2namespace(value)
#         else:
#             new_value = value
#         setattr(namespace, key, new_value)
#     return namespace

def main( config, logger):
    clients_list, t_transform = load_models( config, logger )     # 获取 模型对象

    fednova_train(  config,
                    clients_list,
                    logger=logger,
                 )
    # logger.info("================== end ========================")



if __name__ == "__main__":
    new_config = init()
    logger_level = "INFO"
    logger = get_logger("col",
                        new_config.save_dir,
                        logger_level,
                        f"log_{new_config.name}.txt")  # 收为己用
    main(new_config, logger)