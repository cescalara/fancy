from pathlib import Path
from pkg_resources import resource_filename


def get_path_to_energy_approx_tables(file_name: str) -> Path:

    file_path = resource_filename(
        "fancy", "propagation/energy_loss_tables/%s" % file_name
    )

    return Path(file_path)


def get_available_energy_approx_tables():

    config_path = resource_filename("fancy", "propagation/energy_loss_tables")

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
        "fancy", "propagation/gmf_lens/%s/lens.cfg" % lens_name
    )

    return Path(lens_path)
