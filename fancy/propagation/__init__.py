import os
import requests
import shutil

def install_gmflens():
    '''Install GMFlens from CRPropa database'''
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "gmf_lens")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    lens_tarball = "JF12full.tgz"
    lens_tarball_url = f"https://www.desy.de/~crpropa/data/magnetic_lenses/{lens_tarball}"
    lens_tarball_path = os.path.join(dir_path, lens_tarball)

    # download url from crpropa database & write output to local tarball
    print(f"installing tarball {lens_tarball} from {lens_tarball_url} to {dir_path}")
    r = requests.get(lens_tarball_url)
    with open(lens_tarball_path, 'wb') as f:
        f.write(r.content)

    # verify that the tarball is properly downloaded
    assert lens_tarball in os.listdir(dir_path), "tarball not properly installed!"

    # unpack tarball and remove tarball
    print(f"unpacking tarball in {lens_tarball_path}")
    shutil.unpack_archive(lens_tarball_path, extract_dir=dir_path)
    os.remove(lens_tarball_path)      # remove tarball

    # verify that the lens is fully loaded
    assert "JF12full_Gamale" in os.listdir(dir_path), "tarball not properly unpacked!"
    assert "lens.cfg" in os.listdir(os.path.join(dir_path, "JF12full_Gamale")), "lens does not contain lens.cfg"

