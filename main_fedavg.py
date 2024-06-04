from utils import  get_logger
from core_fedavg import fedavg_train
from common import load_models,  init


def main( config, logger):

    client_list, t_transform = load_models(config, logger)     # 获取 模型对象


    fedavg_train(   config,
                    client_list,
                    logger=logger,
                 )

if __name__ == "__main__":
    new_config = init()  # 获取参数
    logger_level = "INFO"
    logger = get_logger("col",
                        new_config.save_dir,
                        logger_level,
                        f"log_{new_config.name}.txt")  # 收为己用
    main(new_config, logger)

    # pyyaml
    # pip install scikit-learn