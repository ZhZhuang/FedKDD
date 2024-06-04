from utils import get_logger
from core_fedprox import fedprox_train
from common import load_models,  init


def main( config, logger):

    model_list, t_transform = load_models( config, logger)     # 获取 模型对象

    fedprox_train(   config,
                    model_list,
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
    main( new_config, logger)