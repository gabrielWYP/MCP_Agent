import logging
import threading

class SingletonLogger:
    _instance = None
    _lock = threading.Lock()  # Thread-safe para entornos con concurrencia

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SingletonLogger, cls).__new__(cls)
                cls._instance._configure_logger()
        return cls._instance

    def _configure_logger(self):
        self.logger = logging.getLogger("ForecastingSystem")
        self.logger.setLevel(logging.INFO)
        
        # Evitar añadir múltiples handlers si se llama accidentalmente de nuevo
        if not self.logger.handlers:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [Line: %(lineno)d] - %(message)s'
            )
            
            # Handler para consola
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
    
logger_singleton = SingletonLogger().get_logger()