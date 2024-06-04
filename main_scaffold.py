from utils import count_parameters, get_logger
from core_scaffold import scaffold_train

from common import init, load_models

def main( config, logger):

    model_list, t_transform = load_models( config, logger)     # 获取 模型对象

    scaffold_train( config,
                    model_list,
                    logger=logger,
                    # round= config.round,   # 100,
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