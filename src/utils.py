import pynvml

##############################################################################
##############################################################################
#                  *Fonctions utiles divers et variées*                      #
##############################################################################
##############################################################################


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
