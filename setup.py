from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    try:
        """Read the requirements from a file and return them as a list."""
        with open(file_path, 'r') as file:
            requirements = file.readlines()
        return [req.strip() for req in requirements if req.strip() and not req.startswith('#') and not req == '-e .']

    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No requirements will be installed.")

setup(
    name='NetworkSecurity',
    version='0.1.0',
    author='Alwan Adiuntoro',
    author_email='alwanadiuntoro@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements-dev.txt')
)