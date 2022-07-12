import setuptools
import versioneer

setuptools.setup(
    name="fancy",
    packages=setuptools.find_packages(),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    include_package_data=True,
)
