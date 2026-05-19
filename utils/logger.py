import logging
import warnings

# ==========================================
# GEREKSİZ UYARILARI SUSTURMA
# ==========================================
# Transformers'ın "Token indices sequence length is longer..." uyarılarını gizler
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer.*")

# ==========================================
# RENKLİ KONSOL ÇIKTISI (ANSI KODLARI)
# ==========================================
class Colors:
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

class Log:
    @staticmethod
    def step(msg: str):
        """Ana aşamaları belirtmek için (örn: Model Eğitimi Başlıyor)"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}▶ {msg}{Colors.RESET}")

    @staticmethod
    def substep(msg: str):
        """Alt aşamaları belirtmek için (örn: Özellikler Çıkarılıyor)"""
        print(f"  {Colors.BLUE}→{Colors.RESET} {msg}")

    @staticmethod
    def success(msg: str):
        """Başarılı işlemleri belirtmek için"""
        print(f"\n{Colors.GREEN}{Colors.BOLD}✅ {msg}{Colors.RESET}")

    @staticmethod
    def warning(msg: str):
        """Uyarıları belirtmek için"""
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ {msg}{Colors.RESET}")

    @staticmethod
    def error(msg: str):
        """Hataları belirtmek için"""
        print(f"{Colors.RED}{Colors.BOLD}❌ {msg}{Colors.RESET}")

    @staticmethod
    def info(msg: str):
        """Standart bilgilendirme mesajları"""
        print(f"{Colors.DIM}{msg}{Colors.RESET}")
