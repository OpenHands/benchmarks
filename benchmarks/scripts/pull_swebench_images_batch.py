#!/usr/bin/env python3
"""
Pull SWE-Bench Docker images in batches to avoid Docker Hub rate limits.

Usage:
    # Pull all images for SWE-bench_Lite test split
    uv run pull_swebench_images_batch.py --dataset princeton-nlp/SWE-bench_Lite --split test

    # Pull with custom batch size and delay
    uv run pull_swebench_images_batch.py --dataset princeton-nlp/SWE-bench_Lite --split test --batch-size 30 --delay 1800

    # Pull only first 50 instances (for testing)
    uv run pull_swebench_images_batch.py --dataset princeton-nlp/SWE-bench_Lite --split test --n-limit 50
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.swe_bench.build_images import collect_unique_base_images
from openhands.sdk import get_logger


logger = get_logger(__name__)


def check_image_exists_locally(image: str) -> bool:
    """Check if Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Error checking if image {image} exists: {e}")
        return False


def pull_image(image: str, max_retries: int = 3) -> bool:
    """
    Pull a Docker image with retry logic.

    Returns:
        True if successful, False otherwise
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"Pulling {image} (attempt {attempt + 1}/{max_retries})...")
            result = subprocess.run(
                ["docker", "pull", image],
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                logger.info(f"✓ Successfully pulled {image}")
                return True
            else:
                logger.warning(f"Failed to pull {image}: {result.stderr}")
                if attempt < max_retries - 1:
                    logger.info("Retrying in 10 seconds...")
                    time.sleep(10)
        except Exception as e:
            logger.error(f"Exception while pulling {image}: {e}")
            if attempt < max_retries - 1:
                time.sleep(10)

    return False


def save_progress(progress_file: Path, pulled_images: set[str]):
    """Save progress to a JSON file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, "w") as f:
        json.dump({"pulled_images": sorted(list(pulled_images))}, f, indent=2)


def load_progress(progress_file: Path) -> set[str]:
    """Load progress from a JSON file."""
    if not progress_file.exists():
        return set()

    try:
        with open(progress_file, "r") as f:
            data = json.load(f)
            return set(data.get("pulled_images", []))
    except Exception as e:
        logger.warning(f"Could not load progress file: {e}")
        return set()


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Pull SWE-Bench Docker images in batches"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Dataset name (default: princeton-nlp/SWE-bench_Lite)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--n-limit",
        type=int,
        default=None,
        help="Limit number of instances to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of images to pull before waiting (default: 50)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=3600,
        help="Seconds to wait between batches (default: 3600 = 1 hour)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per image (default: 3)",
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default=None,
        help="File to track progress (default: .cache/pull_progress_{dataset}_{split}.json)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist locally",
    )

    args = parser.parse_args(argv)

    # Set up progress file
    if args.progress_file:
        progress_file = Path(args.progress_file)
    else:
        dataset_slug = args.dataset.replace("/", "_").replace("-", "_")
        progress_file = Path(f".cache/pull_progress_{dataset_slug}_{args.split}.json")

    # Load progress
    pulled_images = load_progress(progress_file)
    if pulled_images:
        logger.info(
            f"Resuming from progress file: {len(pulled_images)} images already pulled"
        )

    # Collect all unique base images needed
    logger.info(f"Collecting unique base images from {args.dataset} ({args.split})...")
    try:
        base_images = collect_unique_base_images(
            dataset=args.dataset,
            split=args.split,
            n_limit=args.n_limit,
        )
    except Exception as e:
        logger.error(f"Failed to collect base images: {e}")
        return 1

    logger.info(f"Found {len(base_images)} unique images")

    # Filter out already pulled images
    images_to_pull = [img for img in base_images if img not in pulled_images]

    if args.skip_existing:
        # Further filter by checking local Docker images
        logger.info("Checking which images already exist locally...")
        images_to_pull_filtered = []
        for img in images_to_pull:
            if check_image_exists_locally(img):
                logger.info(f"Skipping {img} (already exists locally)")
                pulled_images.add(img)
            else:
                images_to_pull_filtered.append(img)
        images_to_pull = images_to_pull_filtered

    if not images_to_pull:
        logger.info("All images already pulled!")
        return 0

    logger.info(f"Need to pull {len(images_to_pull)} images")
    logger.info(f"Batch size: {args.batch_size}, Delay between batches: {args.delay}s")

    # Pull images in batches
    failed_images = []

    for idx, image in enumerate(images_to_pull, start=1):
        logger.info(f"\n[{idx}/{len(images_to_pull)}] Processing {image}")

        if pull_image(image, max_retries=args.max_retries):
            pulled_images.add(image)
            save_progress(progress_file, pulled_images)
        else:
            logger.error(f"✗ Failed to pull {image} after {args.max_retries} attempts")
            failed_images.append(image)

        # Wait between batches to avoid rate limit
        if idx % args.batch_size == 0 and idx < len(images_to_pull):
            logger.info(f"\n{'=' * 70}")
            logger.info(
                f"Pulled {args.batch_size} images ({idx}/{len(images_to_pull)} total)"
            )
            logger.info(
                f"Waiting {args.delay} seconds to avoid Docker Hub rate limit..."
            )
            logger.info("Press Ctrl+C to stop (progress will be saved)")
            logger.info(f"{'=' * 70}\n")

            try:
                time.sleep(args.delay)
            except KeyboardInterrupt:
                logger.info("\n\nInterrupted by user. Progress has been saved.")
                logger.info("Resume by running the same command again.")
                return 1

    # Final summary
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 70}")
    logger.info(f"Total images: {len(base_images)}")
    logger.info(f"Successfully pulled: {len(pulled_images)}")
    logger.info(f"Failed: {len(failed_images)}")

    if failed_images:
        logger.info("\nFailed images:")
        for img in failed_images:
            logger.info(f"  - {img}")
        logger.info("\nYou can retry failed images by running the command again.")
        return 1
    else:
        logger.info("\n✓ All images pulled successfully!")
        # Clean up progress file on success
        if progress_file.exists():
            progress_file.unlink()
        return 0


if __name__ == "__main__":
    sys.exit(main())
