import logging
import os
import datetime
from logging.handlers import RotatingFileHandler

def setup_logging(level=logging.DEBUG):
    """
    設定應用程式的日誌記錄器。

    Args:
        level (int): 日誌記錄的最低等級 (e.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)。
    """
    # 確保日誌目錄存在
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 根據目前時間生成日誌檔案名稱
    now = datetime.datetime.now()
    log_file = f"app_{now.strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_file)

    # 取得根記錄器
    logger = logging.getLogger()
    logger.setLevel(level)

    # 避免重複添加處理器
    if not logger.handlers:
        # 建立控制台處理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        console_handler.encoding = 'utf-8'

        # 建立檔案處理器 (輪替檔案，最大 1MB，保留 5 個檔案)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=1024 * 1024, backupCount=50, encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 將處理器添加到記錄器
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # 記錄起始訊息
    logging.info("******************** 程式開始執行 ********************")
    logging.info("日誌記錄器已設定完成")
    return logger

if __name__ == "__main__":
    # 範例用法
    setup_logging()
    logging.debug("這是一條除錯訊息")
    logging.info("這是一條資訊訊息")
    logging.warning("這是一條警告訊息")
    logging.error("這是一條錯誤訊息")
    logging.critical("這是一條嚴重錯誤訊息")