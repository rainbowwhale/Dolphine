"""CUDA setup utilities for handling multiple CUDA installations.

This module provides functionality to detect and configure the CUDA path
before importing CuPy, which helps when multiple CUDA versions are installed
on the same system.
"""

import os
import sys
from pathlib import Path


def find_cuda_paths():
    """Find all existing CUDA installation paths on the system.
    
    Returns:
        list of Path: List of Path objects pointing to existing CUDA installations.
    """
    cuda_paths = []
    
    # Check common CUDA installation directories
    common_locations = [
        '/usr/local/cuda',
        '/usr/local/cuda-*',
        '/opt/cuda',
        '/opt/cuda-*',
    ]
    
    for pattern in common_locations:
        if '*' in pattern:
            # Use glob to find versioned CUDA installations
            base_dir = str(Path(pattern).parent)
            pattern_name = Path(pattern).name
            if os.path.exists(base_dir):
                import glob
                for path in glob.glob(str(Path(base_dir) / pattern_name)):
                    if os.path.isdir(path):
                        cuda_paths.append(Path(path))
        else:
            path = Path(pattern)
            if path.exists() and path.is_dir():
                cuda_paths.append(path)
    
    # Check if CUDA_HOME is already set
    if 'CUDA_HOME' in os.environ:
        cuda_home = Path(os.environ['CUDA_HOME'])
        if cuda_home.exists() and cuda_home not in cuda_paths:
            cuda_paths.insert(0, cuda_home)
    
    # Check if CUDA_PATH is already set
    if 'CUDA_PATH' in os.environ:
        cuda_path = Path(os.environ['CUDA_PATH'])
        if cuda_path.exists() and cuda_path not in cuda_paths:
            cuda_paths.insert(0, cuda_path)
    
    return cuda_paths


def get_cuda_version(cuda_path):
    """Get CUDA version from a CUDA installation path.
    
    Args:
        cuda_path (Path): Path to CUDA installation.
        
    Returns:
        tuple of int or None: (major, minor) version tuple, or None if version cannot be determined.
    """
    version_file = cuda_path / 'version.txt'
    version_json = cuda_path / 'version.json'
    
    # Try version.txt first (older CUDA versions)
    if version_file.exists():
        try:
            with open(version_file, 'r') as f:
                content = f.read().strip()
                # Format is usually "CUDA Version X.Y.Z"
                if 'CUDA Version' in content:
                    version_str = content.split('CUDA Version')[1].strip().split()[0]
                    parts = version_str.split('.')
                    return (int(parts[0]), int(parts[1]))
        except Exception:
            pass
    
    # Try version.json (newer CUDA versions)
    if version_json.exists():
        try:
            import json
            with open(version_json, 'r') as f:
                data = json.load(f)
                if 'cuda' in data and 'version' in data['cuda']:
                    version_str = data['cuda']['version']
                    parts = version_str.split('.')
                    return (int(parts[0]), int(parts[1]))
        except Exception:
            pass
    
    # Try to parse from directory name (e.g., cuda-11.2)
    dir_name = cuda_path.name
    if 'cuda-' in dir_name:
        try:
            version_str = dir_name.split('cuda-')[1]
            parts = version_str.split('.')
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
        except Exception:
            pass
    
    return None


def select_best_cuda_path(cuda_paths):
    """Select the best CUDA path from available options.
    
    Prefers the newest version, or the one already set in environment variables.
    
    Args:
        cuda_paths (list): List of Path objects pointing to CUDA installations.
        
    Returns:
        Path or None: Selected CUDA path, or None if no valid path found.
    """
    if not cuda_paths:
        return None
    
    # If CUDA_PATH or CUDA_HOME is already set and valid, prefer it
    for env_var in ['CUDA_PATH', 'CUDA_HOME']:
        if env_var in os.environ:
            env_path = Path(os.environ[env_var])
            if env_path in cuda_paths:
                return env_path
    
    # Sort by version (newest first)
    versioned_paths = []
    for path in cuda_paths:
        version = get_cuda_version(path)
        if version:
            versioned_paths.append((version, path))
    
    if versioned_paths:
        versioned_paths.sort(reverse=True)
        return versioned_paths[0][1]
    
    # Fallback to first available path
    return cuda_paths[0]


def _add_to_ld_library_path(lib_path):
    """Add a library path to LD_LIBRARY_PATH if not already present.
    
    Args:
        lib_path (str): Library path to add.
    """
    if 'LD_LIBRARY_PATH' in os.environ:
        if lib_path not in os.environ['LD_LIBRARY_PATH'].split(':'):
            os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = lib_path


def setup_cuda_path():
    """Detect and configure CUDA path for CuPy.
    
    This function should be called before importing CuPy when multiple CUDA
    versions might be installed on the system. It sets the CUDA_PATH and
    related environment variables to help CuPy find the correct CUDA libraries.
    
    Returns:
        bool: True if CUDA path was successfully configured, False otherwise.
    """
    # If CUDA_PATH is already set and points to a valid directory, trust it
    if 'CUDA_PATH' in os.environ:
        cuda_path = Path(os.environ['CUDA_PATH'])
        if cuda_path.exists() and (cuda_path / 'lib64').exists():
            # Also set LD_LIBRARY_PATH to help with runtime linking
            lib_path = str(cuda_path / 'lib64')
            _add_to_ld_library_path(lib_path)
            return True
    
    # Find all CUDA installations
    cuda_paths = find_cuda_paths()
    
    if not cuda_paths:
        # No CUDA found, let CuPy handle it (might fail or use a system default)
        return False
    
    # Select the best CUDA path
    selected_path = select_best_cuda_path(cuda_paths)
    
    if selected_path:
        # Set environment variables
        os.environ['CUDA_PATH'] = str(selected_path)
        os.environ['CUDA_HOME'] = str(selected_path)
        
        # Add CUDA lib directory to LD_LIBRARY_PATH
        lib_path = str(selected_path / 'lib64')
        if os.path.exists(lib_path):
            _add_to_ld_library_path(lib_path)
        
        return True
    
    return False


# Automatically setup CUDA path when this module is imported.
# This ensures CUDA is configured before CuPy is imported in solver_core.py.
# The auto-setup is designed to be safe: it respects existing CUDA_PATH settings
# and only configures the environment if needed. For testing or manual control,
# you can call setup_cuda_path() directly after importing this module.
_cuda_configured = setup_cuda_path()
