"""
Command Line Interface for Fed-ML-Lib
=====================================

This module provides a command-line interface for running federated learning experiments.

Usage:
    fed-ml-train --config experiments/config.yaml
    fed-ml-train --model vgg16 --dataset PILL --clients 5 --rounds 10
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from .config import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig
from .utils import get_device


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Fed-ML-Lib: Federated Learning with Classical and Quantum Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with configuration file
  fed-ml-train --config experiments/quantum_federated.yaml
  
  # Run with command line arguments
  fed-ml-train --model vgg16 --dataset PILL --clients 5 --rounds 10
  
  # Quantum federated learning
  fed-ml-train --model hybrid_cnn_qnn --dataset PILL --clients 3 --rounds 5 --batch-size 8
  
  # Non-IID data partitioning
  fed-ml-train --model vgg16 --dataset PILL --partition non_iid --clients 5
        """
    )
    
    # Configuration file
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to YAML/JSON configuration file"
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", "-m",
        type=str,
        choices=["vgg16", "hybrid_cnn_qnn", "quantum", "vqc"],
        default="vgg16",
        help="Model type to use (default: vgg16)"
    )
    model_group.add_argument(
        "--qubits",
        type=int,
        default=4,
        help="Number of qubits for quantum models (default: 4)"
    )
    model_group.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of variational layers for quantum models (default: 2)"
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--dataset", "-d",
        type=str,
        default="PILL",
        help="Dataset name (default: PILL)"
    )
    data_group.add_argument(
        "--data-path",
        type=str,
        default="./data/",
        help="Path to data directory (default: ./data/)"
    )
    data_group.add_argument(
        "--partition",
        type=str,
        choices=["iid", "non_iid", "dirichlet"],
        default="iid",
        help="Data partitioning strategy (default: iid)"
    )
    data_group.add_argument(
        "--resize",
        type=int,
        help="Resize images to this size (default: 224 for classical, 64 for quantum)"
    )
    
    # Training arguments
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--clients", "-n",
        type=int,
        default=5,
        help="Number of federated clients (default: 5)"
    )
    train_group.add_argument(
        "--rounds", "-r",
        type=int,
        default=10,
        help="Number of federated rounds (default: 10)"
    )
    train_group.add_argument(
        "--local-epochs",
        type=int,
        default=2,
        help="Local epochs per round (default: 2)"
    )
    train_group.add_argument(
        "--batch-size", "-b",
        type=int,
        help="Batch size (default: 16 for classical, 8 for quantum)"
    )
    train_group.add_argument(
        "--learning-rate", "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    train_group.add_argument(
        "--lr-strategy",
        type=str,
        choices=["uniform", "split", "adaptive"],
        default="uniform",
        help="Learning rate strategy (default: uniform)"
    )
    train_group.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (default: auto)"
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default="./results/",
        help="Output directory for results (default: ./results/)"
    )
    output_group.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )
    output_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Check if data path exists
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data path '{args.data_path}' does not exist")
        sys.exit(1)
    
    # Check if config file exists (if provided)
    if args.config and not os.path.exists(args.config):
        print(f"❌ Error: Configuration file '{args.config}' does not exist")
        sys.exit(1)
    
    # Validate quantum model parameters
    if args.model in ["hybrid_cnn_qnn", "quantum", "vqc"]:
        if args.qubits < 1 or args.qubits > 20:
            print(f"❌ Error: Number of qubits must be between 1 and 20, got {args.qubits}")
            sys.exit(1)
        
        if args.layers < 1 or args.layers > 10:
            print(f"❌ Error: Number of layers must be between 1 and 10, got {args.layers}")
            sys.exit(1)
    
    # Validate training parameters
    if args.clients < 1:
        print(f"❌ Error: Number of clients must be at least 1, got {args.clients}")
        sys.exit(1)
    
    if args.rounds < 1:
        print(f"❌ Error: Number of rounds must be at least 1, got {args.rounds}")
        sys.exit(1)


def create_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Create experiment configuration from command line arguments."""
    # Set defaults based on model type
    is_quantum = args.model in ["hybrid_cnn_qnn", "quantum", "vqc"]
    
    # Default batch size and image size
    if args.batch_size is None:
        args.batch_size = 8 if is_quantum else 16
    
    if args.resize is None:
        args.resize = 64 if is_quantum else 224
    
    # Model configuration
    model_params = {}
    if is_quantum:
        model_params = {
            "n_qubits": args.qubits,
            "n_layers": args.layers
        }
        if args.model == "quantum":
            model_params["input_size"] = 784  # For flattened images
        elif args.model == "vqc":
            model_params["input_size"] = 4
    
    model_config = ModelConfig(
        model_type=args.model,
        model_params=model_params
    )
    
    # Data configuration
    data_config = DataConfig(
        dataset_name=args.dataset,
        data_path=args.data_path,
        partition_strategy=args.partition,
        resize=args.resize
    )
    
    # Training configuration
    training_config = TrainingConfig(
        num_clients=args.clients,
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        learning_rate_strategy=args.lr_strategy,
        device=args.device,
        seed=args.seed,
        save_results=None if args.no_save else args.output,
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=max(1, args.clients // 2),
        min_evaluate_clients=max(1, args.clients // 2),
        min_available_clients=max(1, args.clients // 2)
    )
    
    return ExperimentConfig(
        model=model_config,
        data=data_config,
        training=training_config
    )


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Print banner
    print("🚀 Fed-ML-Lib: Federated Learning with Classical and Quantum Models")
    print("=" * 70)
    
    # Check dependencies
    try:
        import torch
        import flwr
        print("✅ All required packages are available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please install with: pip install fed-ml-lib[all]")
        sys.exit(1)
    
    # Validate arguments
    validate_args(args)
    
    # Create configuration
    if args.config:
        print(f"📄 Loading configuration from: {args.config}")
        config = ExperimentConfig.from_yaml(args.config)
    else:
        print("⚙️ Creating configuration from command line arguments")
        config = create_config_from_args(args)
    
    # Print configuration summary
    print(f"\n📊 Experiment Configuration:")
    print(f"   Model: {config.model.model_type}")
    print(f"   Dataset: {config.data.dataset_name}")
    print(f"   Clients: {config.training.num_clients}")
    print(f"   Rounds: {config.training.num_rounds}")
    print(f"   Partition: {config.data.partition_strategy}")
    print(f"   Device: {get_device(config.training.device)}")
    
    if config.model.model_type in ["hybrid_cnn_qnn", "quantum", "vqc"]:
        print(f"   Qubits: {config.model.model_params.get('n_qubits', 'N/A')}")
        print(f"   Layers: {config.model.model_params.get('n_layers', 'N/A')}")
    
    # Run experiment
    try:
        from examples.federated_learning_example import run_federated_experiment
        
        print(f"\n🌐 Starting federated learning experiment...")
        results = run_federated_experiment(config)
        
        print(f"\n✅ Experiment completed successfully!")
        print(f"🎯 Final Test Accuracy: {results['final_accuracy']:.4f}")
        
        if config.training.save_results:
            print(f"💾 Results saved to: {config.training.save_results}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 