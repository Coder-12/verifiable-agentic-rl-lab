import argparse
import base64
import os
from pathlib import Path
from typing import Any

import google_crc32c
from google.cloud import storage
from google.cloud.storage.exceptions import DataCorruption


# ============================================================
# CHECKSUM UTILITIES
# ============================================================

def parse_args() -> argparse.Namespace:
    """
    Parse upload configuration without hardcoding credentials in source.

    CLI arguments take priority over environment variables. Supported env vars:
    - GOOGLE_APPLICATION_CREDENTIALS or GCS_SERVICE_ACCOUNT_KEY
    - GCS_BUCKET
    - GCS_LOCAL_FOLDER
    - GCS_PREFIX
    """
    parser = argparse.ArgumentParser(
        description="Upload a local directory to GCS with CRC32C verification."
    )
    parser.add_argument(
        "--key-path",
        default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        or os.getenv("GCS_SERVICE_ACCOUNT_KEY"),
        help="Path to the service-account JSON key.",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("GCS_BUCKET"),
        help="Target GCS bucket name.",
    )
    parser.add_argument(
        "--local-folder",
        default=os.getenv("GCS_LOCAL_FOLDER", "reports"),
        help="Local directory to upload.",
    )
    parser.add_argument(
        "--gcs-folder",
        default=os.getenv("GCS_PREFIX", "countdown_rlvr/reports"),
        help="Target GCS object prefix.",
    )
    args = parser.parse_args()

    if not args.key_path:
        raise ValueError(
            "Missing service-account key path. Pass --key-path or set "
            "GOOGLE_APPLICATION_CREDENTIALS / GCS_SERVICE_ACCOUNT_KEY."
        )
    if not args.bucket:
        raise ValueError("Missing GCS bucket. Pass --bucket or set GCS_BUCKET.")

    return args

def calculate_crc32c(
    file_path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """
    Calculate a file's CRC32C checksum and return it in the same
    Base64-encoded format used by Google Cloud Storage metadata.

    Args:
        file_path:
            Path to the local file.

        chunk_size:
            Number of bytes read at a time. The default is 8 MiB,
            avoiding loading large model files fully into memory.

    Returns:
        Base64-encoded CRC32C checksum.
    """
    checksum = google_crc32c.Checksum()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            checksum.update(chunk)

    return base64.b64encode(checksum.digest()).decode("utf-8")


# ============================================================
# LOCAL FILE DISCOVERY
# ============================================================

def discover_local_files(
    local_folder: Path,
) -> dict[str, dict[str, Any]]:
    """
    Recursively discover every file under local_folder.

    The dictionary key is the relative POSIX path, which is also
    used to construct the corresponding GCS object name.

    Example:
        tokenizer/config.json
        checkpoint-200/model.safetensors
    """
    if not local_folder.exists():
        raise FileNotFoundError(
            f"Local folder does not exist: {local_folder.resolve()}"
        )

    if not local_folder.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but received: {local_folder.resolve()}"
        )

    local_files: dict[str, dict[str, Any]] = {}

    for local_file in sorted(local_folder.rglob("*")):
        if not local_file.is_file():
            continue

        relative_path = local_file.relative_to(local_folder).as_posix()

        local_files[relative_path] = {
            "local_path": local_file,
            "size": local_file.stat().st_size,
            "crc32c": None,
        }

    if not local_files:
        raise RuntimeError(
            f"No files were found under: {local_folder.resolve()}"
        )

    return local_files


# ============================================================
# UPLOAD
# ============================================================

def upload_directory(
    bucket: storage.Bucket,
    local_folder: Path,
    gcs_folder: str,
    local_files: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Upload every discovered local file while preserving the
    complete nested directory structure.

    Each upload uses checksum='crc32c', allowing the Google Cloud
    Storage client to validate transfer integrity automatically.

    After each upload, this function independently compares the
    local checksum and size against the uploaded object's metadata.
    """
    normalized_gcs_folder = gcs_folder.strip("/")

    upload_results: list[dict[str, Any]] = []
    failures: list[str] = []

    total_files = len(local_files)

    for index, (relative_path, metadata) in enumerate(
        local_files.items(),
        start=1,
    ):
        local_path: Path = metadata["local_path"]
        local_size: int = metadata["size"]

        # Calculate independently before upload.
        local_crc32c = calculate_crc32c(local_path)
        metadata["crc32c"] = local_crc32c

        blob_name = f"{normalized_gcs_folder}/{relative_path}"
        blob = bucket.blob(blob_name)

        print(
            f"[{index}/{total_files}] Uploading:\n"
            f"  Local: {local_path}\n"
            f"  GCS:   gs://{bucket.name}/{blob_name}"
        )

        try:
            # The library computes and verifies CRC32C during upload.
            # DataCorruption is raised if client/server values differ.
            blob.upload_from_filename(
                filename=str(local_path),
                checksum="crc32c",
            )

            # Explicitly reload server-side object metadata.
            blob.reload()

            remote_size = int(blob.size or 0)
            remote_crc32c = blob.crc32c

            size_matches = local_size == remote_size
            checksum_matches = local_crc32c == remote_crc32c

            if not size_matches or not checksum_matches:
                mismatch_details = (
                    f"Verification failed for {relative_path}: "
                    f"size local={local_size}, remote={remote_size}; "
                    f"CRC32C local={local_crc32c}, "
                    f"remote={remote_crc32c}"
                )
                failures.append(mismatch_details)
                print(f"  ❌ {mismatch_details}\n")
                continue

            upload_results.append(
                {
                    "relative_path": relative_path,
                    "blob_name": blob_name,
                    "local_size": local_size,
                    "remote_size": remote_size,
                    "local_crc32c": local_crc32c,
                    "remote_crc32c": remote_crc32c,
                    "generation": blob.generation,
                    "verified": True,
                }
            )

            print(
                f"  ✅ Verified\n"
                f"  Size:   {local_size:,} bytes\n"
                f"  CRC32C: {local_crc32c}\n"
            )

        except DataCorruption as error:
            failure = (
                f"GCS detected data corruption while uploading "
                f"{relative_path}: {error}"
            )
            failures.append(failure)
            print(f"  ❌ {failure}\n")

        except Exception as error:
            failure = (
                f"Upload failed for {relative_path}: "
                f"{type(error).__name__}: {error}"
            )
            failures.append(failure)
            print(f"  ❌ {failure}\n")

    if failures:
        formatted_failures = "\n".join(
            f"  - {failure}" for failure in failures
        )

        raise RuntimeError(
            f"\n{len(failures)} file(s) failed upload or verification:\n"
            f"{formatted_failures}"
        )

    return upload_results


# ============================================================
# FINAL FOLDER-LEVEL VERIFICATION
# ============================================================

def verify_uploaded_directory(
    bucket: storage.Bucket,
    gcs_folder: str,
    local_files: dict[str, dict[str, Any]],
) -> None:
    """
    Perform a final independent comparison of the entire local
    directory against all GCS objects under the target prefix.

    This catches:
      - Missing uploaded files
      - Unexpected remote files
      - File-size mismatches
      - CRC32C checksum mismatches
    """
    normalized_gcs_folder = gcs_folder.strip("/")
    prefix = f"{normalized_gcs_folder}/"

    remote_files: dict[str, dict[str, Any]] = {}

    for blob in bucket.list_blobs(prefix=prefix):
        # Ignore a possible zero-byte folder-marker object.
        if blob.name == prefix:
            continue

        relative_path = blob.name[len(prefix):]

        if not relative_path:
            continue

        remote_files[relative_path] = {
            "size": int(blob.size or 0),
            "crc32c": blob.crc32c,
            "generation": blob.generation,
        }

    local_paths = set(local_files)
    remote_paths = set(remote_files)

    missing_files = sorted(local_paths - remote_paths)
    unexpected_files = sorted(remote_paths - local_paths)

    size_mismatches: list[str] = []
    checksum_mismatches: list[str] = []

    for relative_path in sorted(local_paths & remote_paths):
        local_metadata = local_files[relative_path]
        remote_metadata = remote_files[relative_path]

        local_size = local_metadata["size"]
        remote_size = remote_metadata["size"]

        local_crc32c = local_metadata["crc32c"]
        remote_crc32c = remote_metadata["crc32c"]

        if local_size != remote_size:
            size_mismatches.append(
                f"{relative_path}: "
                f"local={local_size:,} bytes, "
                f"remote={remote_size:,} bytes"
            )

        if local_crc32c != remote_crc32c:
            checksum_mismatches.append(
                f"{relative_path}: "
                f"local={local_crc32c}, "
                f"remote={remote_crc32c}"
            )

    print("\n" + "=" * 72)
    print("FINAL DIRECTORY VERIFICATION")
    print("=" * 72)
    print(f"Local files:              {len(local_files)}")
    print(f"Remote files:             {len(remote_files)}")
    print(f"Missing remote files:     {len(missing_files)}")
    print(f"Unexpected remote files:  {len(unexpected_files)}")
    print(f"Size mismatches:          {len(size_mismatches)}")
    print(f"Checksum mismatches:      {len(checksum_mismatches)}")

    verification_errors: list[str] = []

    if missing_files:
        verification_errors.append(
            "Missing files in GCS:\n"
            + "\n".join(f"  - {path}" for path in missing_files)
        )

    if unexpected_files:
        verification_errors.append(
            "Unexpected files already present in the GCS folder:\n"
            + "\n".join(f"  - {path}" for path in unexpected_files)
        )

    if size_mismatches:
        verification_errors.append(
            "File-size mismatches:\n"
            + "\n".join(f"  - {item}" for item in size_mismatches)
        )

    if checksum_mismatches:
        verification_errors.append(
            "CRC32C checksum mismatches:\n"
            + "\n".join(
                f"  - {item}" for item in checksum_mismatches
            )
        )

    if verification_errors:
        error_report = "\n\n".join(verification_errors)

        print("\n❌ FINAL VERIFICATION FAILED")
        print(error_report)

        raise RuntimeError(
            "The uploaded GCS directory is not fully consistent "
            "with the local directory."
        )

    total_bytes = sum(
        metadata["size"] for metadata in local_files.values()
    )

    print("\n✅ FINAL VERIFICATION PASSED")
    print(f"Verified files: {len(local_files)}")
    print(f"Verified bytes: {total_bytes:,}")
    print(
        f"GCS location: gs://{bucket.name}/"
        f"{normalized_gcs_folder}/"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    key_path = Path(args.key_path)
    local_folder = Path(args.local_folder)
    gcs_folder = args.gcs_folder

    if not key_path.exists():
        raise FileNotFoundError(
            f"Service-account key was not found: {key_path.resolve()}"
        )

    print("Discovering local files recursively...")

    local_files = discover_local_files(local_folder)

    total_size = sum(
        metadata["size"] for metadata in local_files.values()
    )

    print(f"Local folder: {local_folder.resolve()}")
    print(f"Files found:  {len(local_files)}")
    print(f"Total size:   {total_size:,} bytes")
    print()

    client = storage.Client.from_service_account_json(
        str(key_path)
    )
    bucket = client.bucket(args.bucket)

    upload_results = upload_directory(
        bucket=bucket,
        local_folder=local_folder,
        gcs_folder=gcs_folder,
        local_files=local_files,
    )

    print(
        f"\nSuccessfully uploaded and individually verified "
        f"{len(upload_results)} file(s)."
    )

    # Final independent folder-level audit.
    verify_uploaded_directory(
        bucket=bucket,
        gcs_folder=gcs_folder,
        local_files=local_files,
    )


if __name__ == "__main__":
    main()
