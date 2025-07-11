"""Run basic tests for CausalBoundingEngine package.

This script runs a subset of tests that work with minimal dependencies.
"""

import sys
import subprocess
import os


def run_tests():
    """Run basic functionality tests."""
    
    # Change to project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Test files to run (in order of importance)
    test_files = [
        'tests/test_dummy.py',
        'tests/test_core_algorithms.py', 
        'tests/test_scenarios.py::TestDataClass',
        'tests/test_scenarios.py::TestScenarioBase',
        'tests/test_integration.py::TestDataHandling',
    ]
    
    print("🧪 Running CausalBoundingEngine Basic Tests\n")
    print("=" * 50)
    
    results = {}
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        print("-" * 40)
        
        try:
            # Run pytest for this test file
            cmd = [
                sys.executable, '-m', 'pytest', 
                test_file, 
                '-v',
                '--tb=short'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {test_file}: PASSED")
                results[test_file] = 'PASSED'
            else:
                print(f"❌ {test_file}: FAILED")
                print(f"Error output:\n{result.stdout}")
                if result.stderr:
                    print(f"Stderr:\n{result.stderr}")
                results[test_file] = 'FAILED'
                
        except Exception as e:
            print(f"💥 {test_file}: ERROR - {str(e)}")
            results[test_file] = 'ERROR'
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r == 'PASSED')
    failed = sum(1 for r in results.values() if r == 'FAILED')
    errors = sum(1 for r in results.values() if r == 'ERROR')
    
    for test_file, result in results.items():
        emoji = "✅" if result == "PASSED" else "❌" if result == "FAILED" else "💥"
        print(f"{emoji} {test_file}: {result}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"💥 Errors: {errors}")
    
    if failed == 0 and errors == 0:
        print("\n🎉 All basic tests passed! The core package functionality is working.")
    else:
        print(f"\n⚠️  Some tests failed. The package may still work, but check the failures above.")
    
    return failed + errors == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
