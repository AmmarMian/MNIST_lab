"""
Main runner script for MNIST/Fashion-MNIST Lab Solutions
Allows running individual parts or all parts sequentially
"""

import sys
import argparse
import warnings

warnings.filterwarnings("ignore")


def run_part1():
    """Run Part 1: Classical ML"""
    print("\n" + "=" * 80)
    print("PART 1: CLASSICAL MACHINE LEARNING (MLP + SVM)")
    print("=" * 80)

    try:
        from part1_classical_ml_complete import main as part1_main

        part1_main()
    except ImportError:
        print("Error: Could not import part1_classical_ml_complete.py")
        print("Make sure the file exists in the current directory.")
        return False
    except Exception as e:
        print(f"Error running Part 1: {e}")
        return False

    return True


def run_part2():
    """Run Part 2: CNNs"""
    print("\n" + "=" * 80)
    print("PART 2: DEEP LEARNING WITH CNNs")
    print("=" * 80)

    try:
        from part2_cnn_complete import main as part2_main

        part2_main()
    except ImportError:
        print("Error: Could not import part2_cnn_complete.py")
        print("Make sure the file exists in the current directory.")
        return False
    except Exception as e:
        print(f"Error running Part 2: {e}")
        return False

    return True


def run_part3():
    """Run Part 3: Transfer Learning"""
    print("\n" + "=" * 80)
    print("PART 3: TRANSFER LEARNING WITH FASHION-MNIST")
    print("=" * 80)

    try:
        from part3_transfer_complete import main as part3_main

        part3_main()
    except ImportError:
        print("Error: Could not import part3_transfer_complete.py")
        print("Make sure the file exists in the current directory.")
        return False
    except Exception as e:
        print(f"Error running Part 3: {e}")
        return False

    return True


def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = {
        "numpy": "numpy",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "torch": "torch",
        "torchvision": "torchvision",
        "seaborn": "seaborn",
    }

    missing_packages = []

    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall missing packages with:")
        print(f"  pip install {' '.join(missing_packages)}")
        print("\nOr using uv:")
        print(f"  uv add {' '.join(missing_packages)}")
        return False

    return True


def print_system_info():
    """Print system and package information"""
    import torch
    import numpy as np
    import sklearn

    print("\nSystem Information:")
    print("-" * 40)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"NumPy version: {np.__version__}")
    print(f"Scikit-learn version: {sklearn.__version__}")
    print(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA available: No (using CPU)")

    print("-" * 40)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run MNIST/Fashion-MNIST Lab Solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all           # Run all parts
  python main.py --part 1        # Run only Part 1
  python main.py --part 2 3      # Run Parts 2 and 3
  python main.py --info          # Show system information only
        """,
    )

    parser.add_argument(
        "--part", type=int, nargs="+", choices=[1, 2, 3], help="Which part(s) to run"
    )
    parser.add_argument("--all", action="store_true", help="Run all parts sequentially")
    parser.add_argument(
        "--info", action="store_true", help="Show system information only"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot display (useful for remote execution)",
    )

    args = parser.parse_args()

    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    print("All dependencies satisfied ✓")

    # Print system info
    if args.info or not (args.part or args.all):
        print_system_info()
        if args.info:
            return 0

    # Disable plots if requested
    if args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        print("\nPlot display disabled (plots will be saved to files)")

    # Determine which parts to run
    if args.all:
        parts_to_run = [1, 2, 3]
    elif args.part:
        parts_to_run = sorted(set(args.part))
    else:
        # Interactive mode
        print("\nWhich part would you like to run?")
        print("1. Classical ML (MLP from scratch + SVM)")
        print("2. Deep Learning (CNNs with PyTorch)")
        print("3. Transfer Learning (Fashion-MNIST)")
        print("4. All parts")

        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            if choice == "4":
                parts_to_run = [1, 2, 3]
            else:
                parts_to_run = [int(choice)]
        except (ValueError, KeyboardInterrupt):
            print("\nInvalid choice or interrupted. Exiting.")
            return 1

    # Run selected parts
    print(f"\nWill run part(s): {parts_to_run}")

    success = True
    for part in parts_to_run:
        if part == 1:
            success &= run_part1()
        elif part == 2:
            success &= run_part2()
        elif part == 3:
            success &= run_part3()

        if not success:
            print(f"\nError occurred in Part {part}. Stopping execution.")
            break

        if part != parts_to_run[-1]:
            try:
                input(
                    f"\nPart {part} completed. Press Enter to continue to Part {part + 1}..."
                )
            except KeyboardInterrupt:
                print("\nInterrupted by user. Exiting.")
                break

    if success:
        print("\n" + "=" * 80)
        print("ALL SELECTED PARTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)

        # Print summary
        print("\nSummary of what was covered:")
        if 1 in parts_to_run:
            print("- Part 1: Implemented MLP from scratch and compared with SVM")
        if 2 in parts_to_run:
            print("- Part 2: Built CNNs with PyTorch and experimented with optimizers")
        if 3 in parts_to_run:
            print("- Part 3: Applied transfer learning to Fashion-MNIST")

        print("\nNext steps:")
        print("- Review the generated plots and results")
        print("- Experiment with different hyperparameters")
        print("- Try the suggested extensions in the README")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
