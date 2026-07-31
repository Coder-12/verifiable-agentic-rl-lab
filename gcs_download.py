import argparse
import base64
import os
from pathlib import Path
from typing import Any

import google_crc32c
from google.cloud import storage


# ============================================================
# CHECKSUM UTILITIES
# ============================================================

def parse_bool(value: str) -> bool:
    """Parse common environment-style boolean values."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def parse_args() -> argparse.Namespace:
    """
    Parse restore configuration without hardcoding credentials in source.

    CLI arguments take priority over environment variables. Supported env vars:
    - GOOGLE_APPLICATION_CREDENTIALS or GCS_SERVICE_ACCOUNT_KEY
    - GCS_BUCKET
    - GCS_PREFIX
    - GCS_LOCAL_FOLDER
    - GCS_OVERWRITE_EXISTING
    """
    parser = argparse.ArgumentParser(
        description="Download a GCS prefix with CRC32C verification."
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
        help="Source GCS bucket name.",
    )
    parser.add_argument(
        "--gcs-folder",
        default=os.getenv(
            "GCS_PREFIX",
            "countdown_rlvr/models/Qwen2.5-1.5B-GRPO-Smoke-step200",
        ),
        help="Source GCS object prefix.",
    )
    parser.add_argument(
        "--local-folder",
        default=os.getenv(
            "GCS_LOCAL_FOLDER",
            "models/Qwen2.5-1.5B-GRPO-Smoke-step200",
        ),
        help="Local restore directory.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        default=parse_bool(os.getenv("GCS_OVERWRITE_EXISTING", "true")),
        help="Overwrite existing files after verification. Enabled by default.",
    )
    parser.add_argument(
        "--no-overwrite-existing",
        action="store_false",
        dest="overwrite_existing",
        help="Require existing local files to match remote objects exactly.",
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
    Calculate the local file's CRC32C checksum.

    Returns the checksum in the same Base64-encoded representation
    used by Google Cloud Storage object metadata.
    """
    checksum = google_crc32c.Checksum()

    with file_path.open("rb") as file:
        while chunk := file.read(chunk_size):
            checksum.update(chunk)

    return base64.b64encode(
        checksum.digest()
    ).decode("utf-8")


# ============================================================
# GCS FILE DISCOVERY
# ============================================================

def discover_gcs_files(
    bucket: storage.Bucket,
    gcs_folder: str,
) -> dict[str, dict[str, Any]]:
    """
    Discover every GCS object recursively under the given prefix.

    GCS does not have real directories. Nested directory structure
    is represented through '/' characters in object names.

    Returns:
        A dictionary keyed by paths relative to gcs_folder.
    """
    normalized_gcs_folder = gcs_folder.strip("/")
    prefix = f"{normalized_gcs_folder}/"

    gcs_files: dict[str, dict[str, Any]] = {}

    for blob in bucket.list_blobs(prefix=prefix):
        # Ignore a possible folder-marker object.
        if blob.name == prefix:
            continue

        relative_path = blob.name[len(prefix):]

        if not relative_path:
            continue

        # Some tools may create zero-byte object names ending in "/".
        # These are directory markers rather than actual files.
        if blob.name.endswith("/"):
            continue

        gcs_files[relative_path] = {
            "blob": blob,
            "blob_name": blob.name,
            "size": int(blob.size or 0),
            "crc32c": blob.crc32c,
            "generation": blob.generation,
        }

    if not gcs_files:
        raise RuntimeError(
            "No files were found under the GCS location:\n"
            f"gs://{bucket.name}/{prefix}"
        )

    return gcs_files


# ============================================================
# SINGLE-FILE VERIFICATION
# ============================================================

def verify_local_file(
    local_path: Path,
    remote_size: int,
    remote_crc32c: str | None,
) -> tuple[bool, str]:
    """
    Verify one local file against its GCS metadata using:
      1. File size
      2. CRC32C checksum

    Returns:
        (verified, explanation)
    """
    if not local_path.exists():
        return False, "Local file does not exist"

    if not local_path.is_file():
        return False, "Local path is not a regular file"

    local_size = local_path.stat().st_size

    if local_size != remote_size:
        return (
            False,
            f"Size mismatch: local={local_size:,}, "
            f"remote={remote_size:,}",
        )

    if remote_crc32c is None:
        return (
            False,
            "GCS object does not contain a CRC32C checksum",
        )

    local_crc32c = calculate_crc32c(local_path)

    if local_crc32c != remote_crc32c:
        return (
            False,
            f"CRC32C mismatch: local={local_crc32c}, "
            f"remote={remote_crc32c}",
        )

    return (
        True,
        f"size={local_size:,} bytes, CRC32C={local_crc32c}",
    )


# ============================================================
# DIRECTORY DOWNLOAD
# ============================================================

def download_directory(
    bucket: storage.Bucket,
    gcs_folder: str,
    local_folder: Path,
    gcs_files: dict[str, dict[str, Any]],
    overwrite_existing: bool = True,
) -> list[dict[str, Any]]:
    """
    Download all objects under a GCS prefix while preserving their
    nested directory structure.

    Downloads first go to a temporary '.part' file. The final target
    is replaced only after the temporary download succeeds.
    """
    normalized_gcs_folder = gcs_folder.strip("/")

    local_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[dict[str, Any]] = []
    failures: list[str] = []

    total_files = len(gcs_files)

    for index, (relative_path, metadata) in enumerate(
        sorted(gcs_files.items()),
        start=1,
    ):
        blob: storage.Blob = metadata["blob"]
        remote_size: int = metadata["size"]
        remote_crc32c: str | None = metadata["crc32c"]

        local_path = local_folder / Path(relative_path)

        # Recreate any nested folders.
        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[{index}/{total_files}]\n"
            f"  GCS:   gs://{bucket.name}/{blob.name}\n"
            f"  Local: {local_path}"
        )

        try:
            # When overwrite is disabled, retain an existing file only
            # when it already matches the remote object exactly.
            if local_path.exists() and not overwrite_existing:
                verified, explanation = verify_local_file(
                    local_path=local_path,
                    remote_size=remote_size,
                    remote_crc32c=remote_crc32c,
                )

                if verified:
                    print(
                        f"  ✅ Existing file already verified; skipped\n"
                        f"  {explanation}\n"
                    )

                    results.append(
                        {
                            "relative_path": relative_path,
                            "local_path": str(local_path),
                            "status": "existing_verified",
                            "verified": True,
                        }
                    )
                    continue

                raise FileExistsError(
                    f"Local file exists but does not match GCS: "
                    f"{local_path}. Details: {explanation}"
                )

            temporary_path = local_path.with_name(
                f"{local_path.name}.part"
            )

            # Remove an incomplete temporary file from an earlier run.
            temporary_path.unlink(missing_ok=True)

            try:
                # The GCS client performs checksum validation during
                # the transfer when checksum="crc32c" is specified.
                blob.download_to_filename(
                    str(temporary_path),
                    checksum="crc32c",
                )

                # Independently verify the temporary file before moving
                # it into the final destination.
                verified, explanation = verify_local_file(
                    local_path=temporary_path,
                    remote_size=remote_size,
                    remote_crc32c=remote_crc32c,
                )

                if not verified:
                    raise RuntimeError(
                        f"Downloaded file verification failed: "
                        f"{explanation}"
                    )

                # Atomic replacement on the same filesystem.
                temporary_path.replace(local_path)

            except Exception:
                temporary_path.unlink(missing_ok=True)
                raise

            # Verify again using the final destination path.
            final_verified, final_explanation = verify_local_file(
                local_path=local_path,
                remote_size=remote_size,
                remote_crc32c=remote_crc32c,
            )

            if not final_verified:
                raise RuntimeError(
                    f"Final-path verification failed: "
                    f"{final_explanation}"
                )

            results.append(
                {
                    "relative_path": relative_path,
                    "local_path": str(local_path),
                    "remote_size": remote_size,
                    "remote_crc32c": remote_crc32c,
                    "generation": metadata["generation"],
                    "status": "downloaded",
                    "verified": True,
                }
            )

            print(
                f"  ✅ Downloaded and verified\n"
                f"  {final_explanation}\n"
            )

        except Exception as error:
            failure = (
                f"{relative_path}: "
                f"{type(error).__name__}: {error}"
            )

            failures.append(failure)
            print(f"  ❌ {failure}\n")

    if failures:
        formatted_failures = "\n".join(
            f"  - {failure}" for failure in failures
        )

        raise RuntimeError(
            f"{len(failures)} file(s) failed to download "
            f"or verify:\n{formatted_failures}"
        )

    print(
        f"Successfully processed {len(results)} files from:\n"
        f"gs://{bucket.name}/{normalized_gcs_folder}/"
    )

    return results


# ============================================================
# FINAL DIRECTORY-LEVEL AUDIT
# ============================================================

def verify_restored_directory(
    local_folder: Path,
    gcs_files: dict[str, dict[str, Any]],
) -> None:
    """
    Compare the entire restored local directory against the GCS
    prefix.

    Detects:
      - Missing local files
      - Unexpected local files
      - Size mismatches
      - CRC32C checksum mismatches
      - Leftover temporary '.part' files
    """
    local_files: dict[str, Path] = {}

    if local_folder.exists():
        for local_path in local_folder.rglob("*"):
            if not local_path.is_file():
                continue

            relative_path = local_path.relative_to(
                local_folder
            ).as_posix()

            local_files[relative_path] = local_path

    remote_paths = set(gcs_files)
    local_paths = set(local_files)

    missing_local_files = sorted(
        remote_paths - local_paths
    )

    unexpected_local_files = sorted(
        local_paths - remote_paths
    )

    size_mismatches: list[str] = []
    checksum_mismatches: list[str] = []
    temporary_files: list[str] = []

    for relative_path in unexpected_local_files:
        if relative_path.endswith(".part"):
            temporary_files.append(relative_path)

    for relative_path in sorted(
        remote_paths & local_paths
    ):
        local_path = local_files[relative_path]
        remote_metadata = gcs_files[relative_path]

        local_size = local_path.stat().st_size
        remote_size = remote_metadata["size"]

        if local_size != remote_size:
            size_mismatches.append(
                f"{relative_path}: "
                f"local={local_size:,} bytes, "
                f"remote={remote_size:,} bytes"
            )
            continue

        local_crc32c = calculate_crc32c(local_path)
        remote_crc32c = remote_metadata["crc32c"]

        if local_crc32c != remote_crc32c:
            checksum_mismatches.append(
                f"{relative_path}: "
                f"local={local_crc32c}, "
                f"remote={remote_crc32c}"
            )

    print("\n" + "=" * 72)
    print("FINAL RESTORE VERIFICATION")
    print("=" * 72)
    print(f"GCS files:                {len(gcs_files)}")
    print(f"Local files:              {len(local_files)}")
    print(f"Missing local files:      {len(missing_local_files)}")
    print(f"Unexpected local files:   {len(unexpected_local_files)}")
    print(f"Size mismatches:          {len(size_mismatches)}")
    print(f"Checksum mismatches:      {len(checksum_mismatches)}")
    print(f"Temporary files:          {len(temporary_files)}")

    verification_errors: list[str] = []

    if missing_local_files:
        verification_errors.append(
            "Files missing locally:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing_local_files
            )
        )

    if unexpected_local_files:
        verification_errors.append(
            "Unexpected files found locally:\n"
            + "\n".join(
                f"  - {path}"
                for path in unexpected_local_files
            )
        )

    if size_mismatches:
        verification_errors.append(
            "File-size mismatches:\n"
            + "\n".join(
                f"  - {item}"
                for item in size_mismatches
            )
        )

    if checksum_mismatches:
        verification_errors.append(
            "CRC32C checksum mismatches:\n"
            + "\n".join(
                f"  - {item}"
                for item in checksum_mismatches
            )
        )

    if verification_errors:
        print("\n❌ FINAL RESTORE VERIFICATION FAILED")
        print("\n\n".join(verification_errors))

        raise RuntimeError(
            "The restored local directory is not fully "
            "consistent with the GCS directory."
        )

    total_bytes = sum(
        metadata["size"]
        for metadata in gcs_files.values()
    )

    print("\n✅ FINAL RESTORE VERIFICATION PASSED")
    print(f"Verified files: {len(gcs_files)}")
    print(f"Verified bytes: {total_bytes:,}")
    print(f"Local location: {local_folder.resolve()}")


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
            "Service-account key was not found:\n"
            f"{key_path.resolve()}"
        )

    client = storage.Client.from_service_account_json(
        str(key_path)
    )

    bucket = client.bucket(args.bucket)

    print("Discovering files in GCS recursively...")

    gcs_files = discover_gcs_files(
        bucket=bucket,
        gcs_folder=gcs_folder,
    )

    total_size = sum(
        metadata["size"]
        for metadata in gcs_files.values()
    )

    print(
        f"GCS source: gs://{args.bucket}/"
        f"{gcs_folder.strip('/')}/"
    )
    print(f"Files found: {len(gcs_files)}")
    print(f"Total size:  {total_size:,} bytes")
    print(f"Destination: {local_folder.resolve()}")
    print()

    results = download_directory(
        bucket=bucket,
        gcs_folder=gcs_folder,
        local_folder=local_folder,
        gcs_files=gcs_files,
        overwrite_existing=args.overwrite_existing,
    )

    print(
        f"\nDownloaded or validated "
        f"{len(results)} file(s)."
    )

    verify_restored_directory(
        local_folder=local_folder,
        gcs_files=gcs_files,
    )


if __name__ == "__main__":
    main()
