"""Model names -> architecture, training speeds and augmentation."""
from .models import ArchConfig, ConvLSTMForecaster, Blank, CopyLast, count_params

ARCH_D = ArchConfig()  # placeholder until chosen on the validation split
MAX_PARAMS_D = 721_176

REGISTRY = {
    "A":  dict(arch=ArchConfig(), speed_aug=False, sym_aug=False),
    "B":  dict(arch=ArchConfig(), speed_aug=False, sym_aug=True),
    "C":  dict(arch=ArchConfig(), speed_aug=True,  sym_aug=False),
    "D":  dict(arch=ARCH_D,       speed_aug=True,  sym_aug=False),
    "DB": dict(arch=ARCH_D,       speed_aug=True,  sym_aug=True),
}


def build_model(name, arch=None):
    """Learned model by name (arch overrides the registry default); blank / copylast too."""
    if name == "blank":
        return Blank()
    if name == "copylast":
        return CopyLast()
    m = ConvLSTMForecaster(arch or REGISTRY[name]["arch"])
    if name in ("D", "DB"):
        assert count_params(m) <= MAX_PARAMS_D, "model D exceeds 721,176 parameters"
    return m
