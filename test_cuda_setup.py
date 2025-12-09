#!/usr/bin/env python3
"""
Comprehensive test for CUDA setup module.
Tests various scenarios without requiring actual CUDA installation.
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def setup_mock_cuda_installations(base_dir):
    """Create mock CUDA installations for testing."""
    installations = [
        ('cuda-11.2', '11.2.0'),
        ('cuda-11.8', '11.8.0'),
        ('cuda-12.0', '12.0.0'),
    ]
    
    created_paths = []
    for dirname, version in installations:
        cuda_path = base_dir / dirname
        cuda_path.mkdir(parents=True, exist_ok=True)
        
        # Create lib64 directory
        lib_path = cuda_path / 'lib64'
        lib_path.mkdir(exist_ok=True)
        
        # Create version.txt
        version_file = cuda_path / 'version.txt'
        with open(version_file, 'w') as f:
            f.write(f'CUDA Version {version}\n')
        
        created_paths.append(cuda_path)
    
    return created_paths


def test_scenario_1_no_cuda():
    """Test when no CUDA is installed."""
    print("\n" + "=" * 60)
    print("Scenario 1: No CUDA Installed")
    print("=" * 60)
    
    # Clear any CUDA environment variables
    for var in ['CUDA_PATH', 'CUDA_HOME', 'LD_LIBRARY_PATH']:
        if var in os.environ:
            del os.environ[var]
    
    # Reimport to reset state
    import importlib
    import cuda_setup
    importlib.reload(cuda_setup)
    
    paths = cuda_setup.find_cuda_paths()
    print(f"Found {len(paths)} CUDA installations")
    assert len(paths) == 0 or all(not p.exists() or 'local' in str(p) for p in paths), \
        "Should find no CUDA or only system default"
    
    result = cuda_setup.setup_cuda_path()
    print(f"Setup result: {result}")
    print("✓ Test passed: Handles no CUDA gracefully")


def test_scenario_2_predefined_cuda_path():
    """Test when CUDA_PATH is already set."""
    print("\n" + "=" * 60)
    print("Scenario 2: CUDA_PATH Pre-defined")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mock_cuda = base / 'my_cuda'
        lib_dir = mock_cuda / 'lib64'
        lib_dir.mkdir(parents=True)
        
        # Set CUDA_PATH
        os.environ['CUDA_PATH'] = str(mock_cuda)
        
        # Reimport to reset state
        import importlib
        import cuda_setup
        importlib.reload(cuda_setup)
        
        result = cuda_setup.setup_cuda_path()
        print(f"Setup result: {result}")
        print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")
        print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
        
        assert result is True, "Should successfully configure when path is predefined"
        assert os.environ['CUDA_PATH'] == str(mock_cuda), "Should respect predefined path"
        assert str(lib_dir) in os.environ.get('LD_LIBRARY_PATH', ''), \
            "Should add lib64 to LD_LIBRARY_PATH"
        
        print("✓ Test passed: Respects pre-defined CUDA_PATH")
        
        # Cleanup
        del os.environ['CUDA_PATH']
        if 'CUDA_HOME' in os.environ:
            del os.environ['CUDA_HOME']


def test_scenario_3_multiple_versions():
    """Test when multiple CUDA versions are available."""
    print("\n" + "=" * 60)
    print("Scenario 3: Multiple CUDA Versions")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        mock_installations = setup_mock_cuda_installations(base)
        
        # Temporarily modify find_cuda_paths to use our mock installations
        import cuda_setup
        original_find = cuda_setup.find_cuda_paths
        
        def mock_find():
            return mock_installations
        
        cuda_setup.find_cuda_paths = mock_find
        
        try:
            paths = cuda_setup.find_cuda_paths()
            print(f"Found {len(paths)} CUDA installations:")
            for p in paths:
                version = cuda_setup.get_cuda_version(p)
                print(f"  - {p.name}: version {version}")
            
            best = cuda_setup.select_best_cuda_path(paths)
            print(f"\nSelected best: {best.name}")
            
            # Should select the newest version (12.0)
            assert '12.0' in best.name, "Should select newest CUDA version"
            
            print("✓ Test passed: Selects newest CUDA version")
        finally:
            cuda_setup.find_cuda_paths = original_find


def test_scenario_4_version_parsing():
    """Test version parsing from different sources."""
    print("\n" + "=" * 60)
    print("Scenario 4: Version Parsing")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        
        # Test version.txt parsing
        cuda1 = base / 'cuda-test1'
        cuda1.mkdir()
        with open(cuda1 / 'version.txt', 'w') as f:
            f.write('CUDA Version 11.5.0\n')
        
        import cuda_setup
        version = cuda_setup.get_cuda_version(cuda1)
        print(f"Version from version.txt: {version}")
        assert version == (11, 5), f"Expected (11, 5), got {version}"
        
        # Test directory name parsing
        cuda2 = base / 'cuda-10.2'
        cuda2.mkdir()
        version = cuda_setup.get_cuda_version(cuda2)
        print(f"Version from directory name: {version}")
        assert version == (10, 2), f"Expected (10, 2), got {version}"
        
        print("✓ Test passed: Version parsing works correctly")


def main():
    print("=" * 60)
    print("CUDA Setup Module - Comprehensive Test Suite")
    print("=" * 60)
    
    try:
        test_scenario_1_no_cuda()
        test_scenario_2_predefined_cuda_path()
        test_scenario_3_multiple_versions()
        test_scenario_4_version_parsing()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
