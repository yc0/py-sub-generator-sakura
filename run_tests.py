#!/usr/bin/env python3
"""
🧪 Test Runner for Sakura Subtitle Generator

Professional test execution with different test categories and reporting.
"""

import sys
import subprocess
from pathlib import Path
import argparse


def run_tests(test_type="all", verbose=False, coverage=False, gpu=False):
    """Run tests with specified configuration."""
    
    cmd = ["uv", "run", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing", "--cov-report=html"])
    
    # Select test types
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "gpu":
        cmd.extend(["-m", "gpu"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow and not gpu"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "no-download":
        cmd.extend(["-m", "not model_download"])
    
    # GPU-specific handling
    if not gpu:
        cmd.extend(["-m", "not gpu"])
    
    # Add test directory
    cmd.append("tests/")
    
    print(f"🧪 Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n🛑 Tests interrupted by user")
        return 130
    except FileNotFoundError:
        print("❌ Error: 'uv' or 'pytest' not found. Please install uv and ensure pytest is in dependencies.")
        return 1


def main():
    """Main test runner."""
    
    parser = argparse.ArgumentParser(
        description="Run Sakura Subtitle Generator tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  all          Run all tests (default)
  unit         Run only unit tests (fast, no external dependencies)
  gpu          Run only GPU-related tests
  slow         Run slow tests (model loading, downloads)
  fast         Run fast tests (no GPU, no downloads)
  integration  Run integration tests
  no-download  Run tests without model downloads

Examples:
  python run_tests.py                    # Run all fast tests
  python run_tests.py --type unit -v     # Run unit tests with verbose output
  python run_tests.py --type gpu --gpu   # Run GPU tests (requires GPU)
  python run_tests.py --coverage         # Run with coverage report
  python run_tests.py --type slow -v     # Run slow tests (may download models)
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "gpu", "slow", "fast", "integration", "no-download"],
        default="fast",
        help="Type of tests to run (default: fast)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true", 
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--gpu", "-g",
        action="store_true",
        help="Include GPU tests (requires GPU acceleration)"
    )
    
    args = parser.parse_args()
    
    # Show configuration
    print("🌸 Sakura Subtitle Generator Test Suite")
    print("=" * 60)
    print(f"Test type: {args.type}")
    print(f"Verbose: {args.verbose}")
    print(f"Coverage: {args.coverage}")
    print(f"GPU tests: {args.gpu}")
    print()
    
    # Check if pytest is available
    try:
        result = subprocess.run(["uv", "run", "pytest", "--version"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ pytest not available. Installing...")
            subprocess.run(["uv", "add", "--dev", "pytest", "pytest-cov"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error setting up test environment")
        return 1
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=args.verbose, 
        coverage=args.coverage,
        gpu=args.gpu
    )
    
    # Report results
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())