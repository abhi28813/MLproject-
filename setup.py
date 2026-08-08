from setuptools import setup, find_packages

HYPEN_E_DOT = "-e ."

def get_requires(file_path: str) -> list[str]:
    requires = []
    with open(file_path) as f:
        requires = f.readlines()
        requires = [req.replace("\n", "") for req in requires]

        if HYPEN_E_DOT in requires:
            requires.remove(HYPEN_E_DOT)

    return requires
   

'''def get_requires(file_path):
    with open(file_path) as f:
        return f.read().splitlines()'''

setup(
    name="mlproject",
    version="0.0.1",
    author="Abhijith",
    author_email="abhijith@example.com",
    packages=find_packages(),
    install_requires=get_requires("requirements.txt")
)