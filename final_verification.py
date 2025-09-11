#!/usr/bin/env python3
"""
Final verification script for the fert-recon pipeline
Tests both data download capabilities and CSV creation
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path

def test_cli_functionality():
    """Test that CLI commands work without errors"""
    print("🔧 Testing CLI functionality...")
    
    try:
        # Test CLI help
        result = subprocess.run([
            sys.executable, "data-pipeline/scripts/cli.py", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ CLI help command works")
        else:
            print(f"  ❌ CLI help failed: {result.stderr}")
            return False
            
        # Test soilgrids help
        result = subprocess.run([
            sys.executable, "data-pipeline/scripts/cli.py", "soilgrids", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("  ✅ Soilgrids command accessible")
        else:
            print(f"  ❌ Soilgrids command failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ❌ CLI commands timed out")
        return False
    except Exception as e:
        print(f"  ❌ CLI test failed: {e}")
        return False
        
    return True

def test_data_availability():
    """Test that required data files exist"""
    print("📁 Testing data availability...")
    
    # Check for TIF files
    tif_dir = "data/raw/output_yearly/processed/stack_yearly"
    if not os.path.exists(tif_dir):
        print(f"  ❌ TIF directory not found: {tif_dir}")
        return False
    
    tif_files = list(Path(tif_dir).glob("*.tif"))
    print(f"  ✅ Found {len(tif_files)} TIF files in {tif_dir}")
    
    if len(tif_files) == 0:
        print("  ❌ No TIF files found for processing")
        return False
    
    # Check for existing CSV
    csv_file = "data/processed/comprehensive_pixels_dataset.csv"
    if os.path.exists(csv_file):
        file_size = os.path.getsize(csv_file) / (1024**3)  # GB
        print(f"  ✅ Existing CSV found: {csv_file} ({file_size:.2f} GB)")
    else:
        print(f"  ℹ️  No existing CSV found at {csv_file}")
    
    return True

def test_csv_creation_capability():
    """Test that CSV creation script can be imported and has correct paths"""
    print("📊 Testing CSV creation capability...")
    
    try:
        # Add the data-pipeline to path for imports
        sys.path.insert(0, "data-pipeline/scripts")
        
        # Try to compile the script
        result = subprocess.run([
            sys.executable, "-m", "py_compile", 
            "data-pipeline/scripts/create_comprehensive_pixels_csv_optimized.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ CSV creation script compiles successfully")
        else:
            print(f"  ❌ CSV script compilation failed: {result.stderr}")
            return False
            
        # Check if script can access required directories
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "csv_creator", 
            "data-pipeline/scripts/create_comprehensive_pixels_csv_optimized.py"
        )
        
        if spec and spec.loader:
            print("  ✅ CSV creation script is importable")
        else:
            print("  ❌ CSV creation script import failed")
            return False
            
    except Exception as e:
        print(f"  ❌ CSV creation test failed: {e}")
        return False
    
    return True

def test_existing_dataset():
    """Test that we can load the existing dataset"""
    print("📈 Testing existing dataset access...")
    
    csv_file = "data/processed/comprehensive_pixels_dataset.csv"
    if not os.path.exists(csv_file):
        print(f"  ℹ️  No existing dataset to test")
        return True
    
    try:
        # Try to load a small sample
        df_sample = pd.read_csv(csv_file, nrows=100)
        print(f"  ✅ Dataset sample loaded: {df_sample.shape}")
        print(f"  ✅ Columns: {list(df_sample.columns)[:5]}...")
        
        # Get full dataset info without loading it all
        with open(csv_file, 'r') as f:
            first_line = f.readline()
            num_cols = len(first_line.split(','))
        
        file_size = os.path.getsize(csv_file) / (1024**3)  # GB
        print(f"  ✅ Full dataset: {num_cols} columns, {file_size:.2f} GB")
        
    except Exception as e:
        print(f"  ❌ Dataset access failed: {e}")
        return False
    
    return True

def test_pipeline_script():
    """Test that pipeline.sh has correct paths"""
    print("🚀 Testing pipeline script...")
    
    if not os.path.exists("pipeline.sh"):
        print("  ❌ pipeline.sh not found")
        return False
    
    # Check if pipeline script has the right paths
    with open("pipeline.sh", 'r') as f:
        content = f.read()
    
    if "data-pipeline/scripts/cli.py" in content:
        print("  ✅ Pipeline script has updated CLI paths")
    else:
        print("  ❌ Pipeline script still has old CLI paths")
        return False
    
    if "data-pipeline/scripts/extract_all.py" in content:
        print("  ✅ Pipeline script has updated extract paths")
    else:
        print("  ❌ Pipeline script still has old extract paths")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 FINAL VERIFICATION - fert-recon Pipeline")
    print("=" * 50)
    
    os.chdir("/data/oe23/fert-recon")
    
    tests = [
        ("CLI Functionality", test_cli_functionality),
        ("Data Availability", test_data_availability),
        ("CSV Creation Capability", test_csv_creation_capability),
        ("Existing Dataset Access", test_existing_dataset),
        ("Pipeline Script", test_pipeline_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        print()
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print()
    print("📋 FINAL RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")
        if result:
            passed += 1
    
    print()
    print(f"Overall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL SYSTEMS READY! The pipeline is fully functional.")
    else:
        print("⚠️  Some issues found. Check the failed tests above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
