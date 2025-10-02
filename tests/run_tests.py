#!/usr/bin/env python3
"""
Test runner script for document processor tests.
Provides easy way to run all tests with different configurations.
"""
import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

def run_unit_tests():
    """Run unit tests."""
    print("=" * 60)
    print("RUNNING UNIT TESTS")
    print("=" * 60)
    
    # Discover and run unit tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_document_processor.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_integration_tests():
    """Run integration tests."""
    print("\n" + "=" * 60)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 60)
    
    # Discover and run integration tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_integration.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_functional_tests():
    """Run functional tests."""
    print("\n" + "=" * 60)
    print("RUNNING FUNCTIONAL TESTS")
    print("=" * 60)
    
    # Discover and run functional tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_functional.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING ALL TESTS")
    print("=" * 60)
    
    # Run test suites
    unit_result = run_unit_tests()
    functional_result = run_functional_tests()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = unit_result.testsRun + functional_result.testsRun
    total_failures = len(unit_result.failures) + len(functional_result.failures)
    total_errors = len(unit_result.errors) + len(functional_result.errors)
    total_success = total_tests - total_failures - total_errors
    
    print(f"Total tests run: {total_tests}")
    print(f"Unit tests: {unit_result.testsRun}")
    print(f"Functional tests: {functional_result.testsRun}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success rate: {(total_success / total_tests * 100):.1f}%")
    
    print("=" * 60)
    
    return total_failures == 0 and total_errors == 0

def main():
    """Main test runner."""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "unit":
            success = run_unit_tests().wasSuccessful()
        elif test_type == "functional":
            success = run_functional_tests().wasSuccessful()
        elif test_type == "all":
            success = run_all_tests()
        else:
            print(f"Unknown test type: {test_type}")
            print("Usage: python run_tests.py [unit|functional|all]")
            sys.exit(1)
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
