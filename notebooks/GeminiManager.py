import os
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


class GeminiManager_class:
    """
    Singleton para manejar la comunicación con Gemini de forma agnóstica.
    Permite modificar dinámicamente los parámetros sin afectar la instancia.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa el singleton una sola vez"""
        if self._initialized:
            return
        
        self.llm = None
        self.config = {
            "model": os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            "temperature": 0,
            "max_output_tokens": 4096,
            "response_mime_type": 'text/plain'
        }
        self._initialized = True
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Inicializa o reinicializa el cliente de Gemini con configuración actual"""
        self.llm =
            model=self.config["model"],
            temperature=self.config["temperature"],
            api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=self.config["max_output_tokens"],
            response_mime_type=self.config["response_mime_type"]
        )
    
    @classmethod
    def get_instance(cls):
        """Obtiene la instancia única del singleton"""
        if cls._instance is None:
            cls()
        if cls._instance.llm is None:
            cls._instance._initialize_llm() # Llama a la inicialización real
        return cls._instance
    
    def set_config(self, **kwargs):
        
        self.config.update(kwargs)
        self._initialize_llm()
        print(f"Configuración de Gemini actualizada: {self.config}")
    
    def get_config(self):
        """Retorna la configuración actual"""
        return self.config.copy()
    
    def invoke(self, mensaje: str, **temp_params) -> str:
        
        if temp_params:
            # Guarda configuración actual
            backup = self.config.copy()
            
            # Aplica cambios temporales
            self.config.update(temp_params)
            self._initialize_llm()
            
            # Ejecuta la llamada
            result = self.llm.invoke(mensaje).content
            
            # Restaura configuración anterior
            self.config = backup
            self._initialize_llm()
            
            return result
        
        # Si no hay parámetros temporales, usa configuración actual
        return self.llm.invoke(mensaje).content
    
    def reset_config(self):
        """Resetea la configuración a valores por defecto"""
        self.config = {
            "model": os.getenv("LLM_MODEL", "gemini-2.5-flash"),
            "temperature": 0,
            "max_output_tokens": 4096,
            "response_mime_type": 'text/plain'
        }
        self._initialize_llm()
        print("✅ Configuración reseteada a valores por defecto")
    
    @classmethod
    def destroy(cls):
        if cls._instance is not None:
            cls._instance.llm = None
            cls._instance._initialized = False
            cls._instance = None
            print("✅ Singleton GeminiManager destruido")
        else:
            print("⚠️ No hay instancia activa que destruir")
    
    @classmethod
    def reinitialize(cls):
        """
        Destruye la instancia actual y crea una nueva limpia.
        Útil para reiniciar después de errores o cambios drásticos.
        
        Returns:
            GeminiManager: Nueva instancia singleton
            
        Example:
            gemini_manager = GeminiManager_class.reinitialize()
            # Tienes una instancia fresca
        """
        cls.destroy()
        return cls.get_instance()
    
    def __repr__(self):
        """Representación del objeto"""
        return f"GeminiManager(model={self.config['model']}, temp={self.config['temperature']})"
