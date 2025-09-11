#!/usr/bin/env python3
"""Test script to verify imports work correctly"""

import sys
import os

# Add the data-pipeline directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data-pipeline'))

try:
    from config.config import DEFAULT_COUNTRY
    print(f"✅ Config import successful. Default country: {DEFAULT_COUNTRY}")
except ImportError as e:
    print(f"❌ Config import failed: {e}")

try:
    from config.datasets import DICO
    if DICO is None:
        print("✅ Datasets import successful. DICO is None (not initialized yet)")
    else:
        print(f"✅ Datasets import successful. Found {len(DICO)} datasets")
except ImportError as e:
    print(f"❌ Datasets import failed: {e}")

try:
    from shared.utils.geometry import country_bbox
    print("✅ Shared utils geometry import successful")
except ImportError as e:
    print(f"❌ Shared utils geometry import failed: {e}")

try:
    # Skip ee_helpers since it might require authentication
    print("⏭️ Skipping ee_helpers import (might require authentication)")
except ImportError as e:
    print(f"❌ Shared utils ee_helpers import failed: {e}")

print("\nTest completed!")
