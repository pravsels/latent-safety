#!/usr/bin/env python3
"""
Script to filter Hugging Face datasets by robot_type.

For "arx5", directly checks camera data to ensure single-arm setup:
- Has observation.images.front
- Has observation.images.wrist
- Does NOT have observation.images.left_wrist or observation.images.right_wrist

Sample Usage:
    # Test a single dataset
    python scripts/filter_hf_datasets.py --test-dataset villekuosmanen/drop_footbag_into_dice_tower_continuous --robot-type arx5
    
    # Filter all datasets from a user
    python scripts/filter_hf_datasets.py --username villekuosmanen --robot-type arx5 --output arx5_datasets.json
    
    # Filter with quiet mode
    python scripts/filter_hf_datasets.py --username villekuosmanen --robot-type arx5 --output arx5_datasets.json --quiet
"""

import json
from typing import List
from huggingface_hub import list_datasets, HfApi


def _is_single_arm_arx5(has_front: bool, has_wrist: bool, has_left_wrist: bool, has_right_wrist: bool, verbose: bool = False) -> bool:
    """Check if camera configuration indicates single-arm ARX5."""
    if has_front and has_wrist and not has_left_wrist and not has_right_wrist:
        return True
    if verbose:
        print(f"    Not single-arm: front={has_front}, wrist={has_wrist}, left_wrist={has_left_wrist}, right_wrist={has_right_wrist}")
    return False


def check_robot_type(dataset_id: str, target_robot_type: str = "arx5", verbose: bool = False) -> bool:
    """
    Check if dataset matches target robot_type by checking repo file structure.
    
    Args:
        dataset_id: Full dataset ID (e.g., "villekuosmanen/dataset_name")
        target_robot_type: Target robot type to filter by
        verbose: Whether to print debug information
        
    Returns:
        True if robot_type matches and is single-arm (for arx5), False otherwise
    """
    target_lower = target_robot_type.lower()
    
    # Check the repo file structure directly for video directories
    try:
        api = HfApi()
        dataset_info = api.dataset_info(dataset_id)
        
        if not hasattr(dataset_info, 'siblings'):
            if verbose:
                print("    No siblings found in repo")
            return False
        
        siblings = [s.rfilename for s in dataset_info.siblings]
        
        # Check if observation.images.* directories exist
        has_front = any('observation.images.front' in s for s in siblings)
        has_wrist = any('observation.images.wrist' in s for s in siblings)
        has_left_wrist = any('observation.images.left_wrist' in s for s in siblings)
        has_right_wrist = any('observation.images.right_wrist' in s for s in siblings)
        
        if verbose:
            print(f"    Video dirs: front={has_front}, wrist={has_wrist}, left_wrist={has_left_wrist}, right_wrist={has_right_wrist}")
        
        if target_lower == "arx5":
            return _is_single_arm_arx5(has_front, has_wrist, has_left_wrist, has_right_wrist, verbose)
        else:
            # For other robot types, just return True (no camera check)
            return True
            
    except Exception as e:
        if verbose:
            error_msg = str(e)
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            print(f"    Error: {error_msg}")
        return False


def filter_datasets_by_robot_type(
    username: str,
    robot_type: str = "arx5",
    verbose: bool = True
) -> List[str]:
    """
    List all datasets from a user and filter by robot_type.
    
    Args:
        username: Hugging Face username
        robot_type: Target robot type to filter by (default: "arx5")
        verbose: Whether to print progress information
        
    Returns:
        List of dataset IDs that match the filter
    """
    if verbose:
        print(f"Fetching all datasets from user: {username}")
    
    # List all datasets from the user
    try:
        all_datasets = list_datasets(author=username)
        dataset_ids = [ds.id for ds in all_datasets]
    except Exception as e:
        print(f"Error listing datasets: {e}")
        return []
    
    if verbose:
        print(f"Found {len(dataset_ids)} datasets")
        print(f"Filtering for robot_type: '{robot_type}'...")
        print()
    
    matching_datasets = []
    
    for i, dataset_id in enumerate(dataset_ids, 1):
        if verbose:
            print(f"[{i}/{len(dataset_ids)}] Checking {dataset_id}...", end=" ")
        
        match_result = check_robot_type(dataset_id, robot_type, verbose=verbose)
        if match_result:
            matching_datasets.append(dataset_id)
            if verbose:
                print("✓ MATCH")
        else:
            if verbose:
                print("✗")
    
    return matching_datasets


def main():
    """Main function to run the filtering script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter Hugging Face datasets by robot_type"
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Hugging Face username (required unless using --test-dataset)"
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="arx5",
        help="Robot type to filter by (default: arx5)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file to save matching dataset IDs (JSON format)"
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        help="Test a single dataset ID (e.g., 'villekuosmanen/dataset_name')"
    )
    
    args = parser.parse_args()
    
    # Test mode: check a single dataset
    if args.test_dataset:
        print(f"Testing dataset: {args.test_dataset}")
        print("=" * 60)
        result = check_robot_type(args.test_dataset, args.robot_type, verbose=True)
        print("=" * 60)
        if result:
            print("✓ MATCH - Dataset passes filter")
        else:
            print("✗ NO MATCH - Dataset does not pass filter")
        return [args.test_dataset] if result else []
    
    # Normal mode: filter all datasets from user
    if not args.username:
        parser.error("Must provide either --username or --test-dataset")
    
    matching = filter_datasets_by_robot_type(
        username=args.username,
        robot_type=args.robot_type,
        verbose=not args.quiet
    )
    
    print()
    print("=" * 60)
    print(f"Found {len(matching)} dataset(s) with robot_type='{args.robot_type}':")
    print("=" * 60)
    
    if matching:
        for dataset_id in matching:
            print(f"  - {dataset_id}")
    else:
        print("  (none)")
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(matching, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return matching


if __name__ == "__main__":
    main()
