from pathlib import Path
from pkg_resources import resource_filename


def get_path_to_energy_approx_tables(file_name: str) -> Path:

    file_path = resource_filename(
        "fancy", "propagation/energy_loss_tables/%s" % file_name
    )

    return Path(file_path)


def get_available_energy_approx_tables():

    config_path = resource_filename("fancy", "physics/energy_loss/energy_loss_tables")

    paths = list(Path(config_path).rglob("*.h5"))

    files = [p.name for p in paths]

    return files


def get_path_to_stan_file(file_name: str) -> Path:

    file_path = resource_filename("fancy", "interfaces/stan/%s" % file_name)

    return Path(file_path)


def get_path_to_stan_includes() -> Path:

    include_path = resource_filename("fancy", "interfaces/stan")

    return Path(include_path)


def get_path_to_lens(lens_name: str) -> Path:

    lens_path = resource_filename(
        "fancy", "physics/gmf/gmf_lens/%s/lens.cfg" % lens_name
    )

    return Path(lens_path)

def get_path_to_kappa_theta(file_name : str = "kappa_theta_map.pkl") -> Path:

    kappa_theta_path = resource_filename(
        "fancy", "utils/resources/{0:s}".format(file_name)
    )
    return Path(kappa_theta_path)
