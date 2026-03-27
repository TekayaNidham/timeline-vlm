"""
TIME10k Dataset Downloader.

Downloads images from Wikimedia Commons URLs listed in time10k.csv
and organizes them into the year-based directory structure expected by the dataset loader.

Usage:
    python data/download.py --csv data/time10k.csv --output data/TIME10k
    python data/download.py --csv data/time10k.csv --output data/TIME10k --workers 16
"""

import os
import argparse
import csv
import hashlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote

import requests
from tqdm import tqdm


def sanitize_filename(url):
    """Extract and sanitize filename from URL."""
    parsed = urlparse(url)
    filename = unquote(os.path.basename(parsed.path))
    # Remove problematic characters
    filename = filename.replace('%', '_').replace('?', '_').replace('&', '_')
    if not filename or filename == '/':
        filename = hashlib.md5(url.encode()).hexdigest() + '.jpg'
    return filename


def download_image(url, output_path, timeout=30, retries=3):
    """
    Download a single image with retry logic.

    Args:
        url: Image URL
        output_path: Full path to save the image
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        (success: bool, message: str)
    """
    if os.path.exists(output_path):
        return True, "already exists"

    headers = {
        'User-Agent': 'TIME10kDatasetDownloader/1.0 '
                       '(https://github.com/tekayanidham/timeline-vlm; '
                       'research dataset download)'
    }

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout,
                                    stream=True)
            response.raise_for_status()

            # Verify it's an image
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type and 'octet-stream' not in content_type:
                return False, f"not an image: {content_type}"

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True, "downloaded"

        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))  # Backoff
                continue
            return False, str(e)

    return False, "max retries exceeded"


def load_csv(csv_path):
    """
    Load time10k.csv and return list of (url, year, class, object_name) tuples.
    """
    entries = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row['url'].strip()
            year = int(row['groundtruth_year'])
            cls = row['class'].strip().strip('"')
            obj_name = row.get('object_name', '').strip().strip('"')
            entries.append((url, year, cls, obj_name))
    return entries


def download_dataset(csv_path, output_dir, workers=8, skip_existing=True):
    """
    Download the full TIME10k dataset from URLs in the CSV.

    Creates directory structure: output_dir/{year}/{filename}

    Args:
        csv_path: Path to time10k.csv
        output_dir: Root directory for downloaded images
        workers: Number of parallel download threads
        skip_existing: Skip already downloaded images

    Returns:
        Dictionary with download statistics
    """
    entries = load_csv(csv_path)
    print(f"Loaded {len(entries)} entries from {csv_path}")

    # Prepare download tasks
    tasks = []
    for url, year, cls, obj_name in entries:
        filename = sanitize_filename(url)
        output_path = os.path.join(output_dir, str(year), filename)
        tasks.append((url, output_path, year, cls))

    # Check existing
    if skip_existing:
        to_download = [(u, p, y, c) for u, p, y, c in tasks if not os.path.exists(p)]
        existing = len(tasks) - len(to_download)
        print(f"Skipping {existing} already downloaded images")
        print(f"Downloading {len(to_download)} images...")
    else:
        to_download = tasks
        print(f"Downloading {len(to_download)} images...")

    if not to_download:
        print("All images already downloaded!")
        return {'total': len(tasks), 'downloaded': 0, 'existing': len(tasks),
                'failed': 0}

    stats = {'downloaded': 0, 'failed': 0, 'errors': []}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for url, output_path, year, cls in to_download:
            future = executor.submit(download_image, url, output_path)
            futures[future] = (url, output_path, year, cls)

        with tqdm(total=len(to_download), desc="Downloading") as pbar:
            for future in as_completed(futures):
                url, output_path, year, cls = futures[future]
                success, msg = future.result()

                if success:
                    stats['downloaded'] += 1
                else:
                    stats['failed'] += 1
                    stats['errors'].append({'url': url, 'year': year,
                                            'error': msg})

                pbar.update(1)
                pbar.set_postfix(ok=stats['downloaded'], fail=stats['failed'])

    # Summary
    total = len(tasks)
    existing = total - len(to_download)
    print(f"\nDownload complete:")
    print(f"  Total entries:  {total}")
    print(f"  Already existed: {existing}")
    print(f"  Downloaded:     {stats['downloaded']}")
    print(f"  Failed:         {stats['failed']}")

    if stats['errors']:
        error_log = os.path.join(output_dir, 'download_errors.log')
        with open(error_log, 'w') as f:
            for err in stats['errors']:
                f.write(f"{err['url']}\t{err['year']}\t{err['error']}\n")
        print(f"  Error log:      {error_log}")

    return {
        'total': total,
        'existing': existing,
        'downloaded': stats['downloaded'],
        'failed': stats['failed'],
    }


def verify_dataset(csv_path, data_dir):
    """
    Verify downloaded dataset against CSV.

    Returns:
        Dictionary with verification results
    """
    entries = load_csv(csv_path)
    found = 0
    missing = []

    for url, year, cls, obj_name in entries:
        filename = sanitize_filename(url)
        path = os.path.join(data_dir, str(year), filename)
        if os.path.exists(path):
            found += 1
        else:
            missing.append((url, year, cls))

    total = len(entries)
    print(f"Dataset verification:")
    print(f"  Total entries: {total}")
    print(f"  Found: {found} ({100 * found / total:.1f}%)")
    print(f"  Missing: {len(missing)}")

    return {'total': total, 'found': found, 'missing': len(missing),
            'missing_entries': missing}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download TIME10k dataset from Wikimedia Commons URLs')
    parser.add_argument('--csv', type=str, default='data/time10k.csv',
                        help='Path to time10k.csv')
    parser.add_argument('--output', type=str, default='data/TIME10k',
                        help='Output directory for images')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel download threads')
    parser.add_argument('--verify', action='store_true',
                        help='Verify existing dataset instead of downloading')

    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.csv, args.output)
    else:
        download_dataset(args.csv, args.output, workers=args.workers)
