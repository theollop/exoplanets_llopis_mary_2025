import pynvml
import importlib
import torch
import gc

##############################################################################
##############################################################################
#                  *Fonctions utiles divers et variées*                      #
##############################################################################
##############################################################################


def clear_gpu_memory():
    """
    Fonction utilitaire pour nettoyer agressivement la mémoire GPU.
    À utiliser après des opérations coûteuses en mémoire.
    """
    # Force le garbage collection Python
    gc.collect()
    
    # Vide le cache CUDA si disponible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Attendre que toutes les opérations CUDA soient terminées


def get_gpu_memory_info():
    """
    Retourne un dictionnaire avec des informations détaillées sur la mémoire GPU.
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
        
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()
    
    total_mb = info.total / (1024 ** 2)
    used_mb = info.used / (1024 ** 2)
    free_mb = info.free / (1024 ** 2)
    
    return {
        "total_mb": total_mb,
        "used_mb": used_mb,
        "free_mb": free_mb,
        "usage_percent": (used_mb / total_mb) * 100
    }


def get_free_memory():
    """Récupère la mémoire libre du GPU. la renvoie en octets."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    free_memory = info.free  # En octets
    pynvml.nvmlShutdown()
    return free_memory


def get_total_memory():
    """Récupère la mémoire totale du GPU en octets."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = info.total  # En octets
    pynvml.nvmlShutdown()
    return total_memory


def get_used_memory():
    """Récupère la mémoire utilisée du GPU en octets."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used_memory = info.used  # En octets
    pynvml.nvmlShutdown()
    return used_memory


def get_class(path: str):
    """Charge une classe à partir de sa chaîne 'module.submodule.ClassName'. -> utile quand on veut charger une classe depuis un fichier config."""
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
