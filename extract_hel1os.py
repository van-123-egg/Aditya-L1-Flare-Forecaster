"""
extract_hel1os.py — Extract all HEL1OS Level-1 zip files.

For each day folder hel1os/HLS_YYYYMMDD/:
  - Finds all .zip files
  - Extracts each to a folder with the same name (minus .zip) alongside the zip
  - Skips if the folder already exists (idempotent)
  - Handles errors gracefully (prints warning, continues)
  - Prints a summary at the end

Usage:
    python extract_hel1os.py
"""

import os
import zipfile
import time

# ── Configuration ──
HEL1OS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hel1os")

def extract_all():
    """Walk hel1os/ and extract every zip file found."""
    extracted_count = 0
    skipped_count = 0
    failed_count = 0
    failed_files = []

    if not os.path.isdir(HEL1OS_DIR):
        print(f"[ERROR] HEL1OS directory not found: {HEL1OS_DIR}")
        return

    # Find all day folders (HLS_YYYYMMDD)
    day_folders = sorted([
        d for d in os.listdir(HEL1OS_DIR)
        if os.path.isdir(os.path.join(HEL1OS_DIR, d)) and d.startswith("HLS_")
    ])

    print(f"Found {len(day_folders)} day folders in {HEL1OS_DIR}")
    print("=" * 70)

    total_zips = 0

    for day_folder in day_folders:
        day_path = os.path.join(HEL1OS_DIR, day_folder)

        # Find all zip files in this day folder
        zip_files = sorted([
            f for f in os.listdir(day_path)
            if f.lower().endswith(".zip")
        ])

        if not zip_files:
            print(f"\n{day_folder}: No zip files found")
            continue

        total_zips += len(zip_files)
        print(f"\n{day_folder}: {len(zip_files)} zip file(s)")

        for zip_name in zip_files:
            zip_path = os.path.join(day_path, zip_name)
            # The extracted folder name = zip filename without .zip extension
            extract_folder_name = zip_name[:-4]  # strip .zip
            extract_folder_path = os.path.join(day_path, extract_folder_name)

            # Check if already extracted (idempotent)
            if os.path.isdir(extract_folder_path):
                # Verify it's not empty
                has_files = False
                for root, dirs, files in os.walk(extract_folder_path):
                    if files:
                        has_files = True
                        break
                if has_files:
                    print(f"  SKIP  {zip_name} -> folder exists")
                    skipped_count += 1
                    continue
                else:
                    print(f"  REDO  {zip_name} -> folder exists but is empty, re-extracting")

            # Extract
            try:
                t0 = time.time()
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Validate the zip first
                    bad_file = zf.testzip()
                    if bad_file is not None:
                        print(f"  WARN  {zip_name} -> corrupt member: {bad_file}")
                        # Still try to extract what we can

                    zf.extractall(extract_folder_path)

                elapsed = time.time() - t0
                # Count extracted files
                n_files = sum(len(files) for _, _, files in os.walk(extract_folder_path))
                size_mb = os.path.getsize(zip_path) / (1024 * 1024)
                print(f"  OK    {zip_name} ({size_mb:.1f} MB) -> {n_files} files in {elapsed:.1f}s")
                extracted_count += 1

            except zipfile.BadZipFile as e:
                print(f"  FAIL  {zip_name} -> Bad zip file: {e}")
                failed_count += 1
                failed_files.append(zip_name)
            except PermissionError as e:
                print(f"  FAIL  {zip_name} -> Permission denied: {e}")
                failed_count += 1
                failed_files.append(zip_name)
            except Exception as e:
                print(f"  FAIL  {zip_name} -> {type(e).__name__}: {e}")
                failed_count += 1
                failed_files.append(zip_name)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"  Total zip files found:  {total_zips}")
    print(f"  Extracted successfully: {extracted_count}")
    print(f"  Skipped (already done): {skipped_count}")
    print(f"  Failed:                 {failed_count}")

    if failed_files:
        print(f"\n  Failed files:")
        for f in failed_files:
            print(f"    - {f}")

    print(f"\nTotal: {extracted_count + skipped_count + failed_count} processed")
    print("DONE")


if __name__ == "__main__":
    extract_all()
