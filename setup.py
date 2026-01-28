from setuptools import setup, find_packages

setup(
    name='botsort_tracker',
    version='1.0.0',
    description='BoTSORT Tracker only',
    # We explicitly list 'tracker' to ensure we only package the tracker directory
    # and exclude other directories like yolox, fast_reid etc.
    packages=['tracker'],
    # If using find_packages(), we would need to exclude others:
    # packages=find_packages(include=['tracker', 'tracker.*']),
)
