"""
Main module responsible for providing the correct paths
"""

from pathlib import Path


def get_project_root() -> Path:
    """
    Returns the project root path
    ! do not move this function from here !
    """
    return Path(__file__).resolve().parents[2]
