#!/usr/bin/env python3
"""Utility to pre-cache the policy model for faster loading.

This script loads the joblib model and saves it as a pickle cache,
so subsequent loads are much faster (typically 10-100x faster).

Usage:
    python tools/cache_policy_model.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.policy_model import PolicyModelRuntime


def main() -> None:
    model_path = Path("engines/train_eval/model_artifacts/policy_xgb.joblib")
    cache_path = model_path.with_suffix(".pkl")

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return

    print(f"📦 Loading policy model from {model_path}...")
    print("   (This may take 1-2 minutes on first load)")

    try:
        # Load the model using PolicyModelRuntime
        # This will load from joblib and attempt to cache
        runtime = PolicyModelRuntime(model_path=model_path, threshold=0.5)

        if runtime.available:
            print(f"✅ Model loaded successfully!")
            print(f"   Features: {len(runtime.feature_columns)}")
            print(f"   Numeric columns: {len(runtime.numeric_columns)}")

            # Check if cache was created
            if cache_path.exists():
                cache_size_mb = cache_path.stat().st_size / (1024 * 1024)
                print(f"✅ Cache created at {cache_path}")
                print(f"   Cache size: {cache_size_mb:.1f} MB")
                print(f"\n🚀 Next load will be ~10-100x faster!")
            else:
                print(f"⚠️  Cache was not created")
        else:
            print(f"❌ Failed to load model: {runtime.error}")

    except Exception as exc:
        print(f"❌ Error: {exc}")


if __name__ == "__main__":
    main()
