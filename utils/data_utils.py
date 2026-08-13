"""
Data utilities for Med-PRM
==========================
Handles JSON I/O, data validation, and format conversion.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path


def load_json(
    file_path: str,
    encoding: str = "utf-8",
    handle_errors: bool = True
) -> Union[Dict, List, None]:
    """
    Load JSON file with error handling.

    Args:
        file_path: Path to JSON file
        encoding: File encoding (default: utf-8)
        handle_errors: If True, return None on error; if False, raise

    Returns:
        Parsed JSON data or None if error

    Examples:
        >>> data = load_json("dataset.json")
        >>> len(data)
        1000
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return json.load(f)
    except FileNotFoundError:
        if handle_errors:
            print(f"❌ File not found: {file_path}")
            return None
        raise
    except json.JSONDecodeError as e:
        if handle_errors:
            print(f"❌ Invalid JSON in {file_path}: {e}")
            return None
        raise
    except Exception as e:
        if handle_errors:
            print(f"❌ Error loading {file_path}: {e}")
            return None
        raise


def save_json(
    data: Union[Dict, List],
    file_path: str,
    indent: int = 4,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
    create_backup: bool = False
) -> bool:
    """
    Save data to JSON file with optional backup.

    Args:
        data: Data to save
        file_path: Output path
        indent: JSON indentation level
        ensure_ascii: Whether to escape non-ASCII characters
        encoding: File encoding
        create_backup: Create .backup file before overwriting

    Returns:
        True if successful, False otherwise

    Examples:
        >>> data = [{"id": 1, "name": "Test"}]
        >>> save_json(data, "output.json")
        True
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Create backup if file exists
        if create_backup and file_path.exists():
            backup_path = file_path.with_suffix(".backup")
            file_path.rename(backup_path)
            print(f"📦 Backup created: {backup_path}")

        # Write with atomic operation
        temp_path = file_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

        # Atomic rename
        temp_path.replace(file_path)
        print(f"✅ Data saved: {file_path}")
        return True

    except Exception as e:
        print(f"❌ Error saving {file_path}: {e}")
        return False


def load_jsonl(
    file_path: str,
    encoding: str = "utf-8",
    max_lines: Optional[int] = None
) -> List[Dict]:
    """
    Load JSONL file (one JSON object per line).

    Args:
        file_path: Path to JSONL file
        encoding: File encoding
        max_lines: Maximum lines to read (None = all)

    Returns:
        List of parsed JSON objects

    Examples:
        >>> data = load_jsonl("predictions.jsonl")
        >>> len(data)
        5000
    """
    data = []
    line_count = 0

    try:
        with open(file_path, "r", encoding=encoding) as f:
            for line in f:
                if max_lines and line_count >= max_lines:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                    data.append(obj)
                    line_count += 1
                except json.JSONDecodeError as e:
                    print(f"⚠️  Skipping invalid JSON on line {line_count + 1}: {e}")

        print(f"✅ Loaded {len(data)} items from {file_path}")
        return data

    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []


def save_jsonl(
    data: List[Dict],
    file_path: str,
    encoding: str = "utf-8"
) -> bool:
    """
    Save data to JSONL file.

    Args:
        data: List of dictionaries
        file_path: Output path
        encoding: File encoding

    Returns:
        True if successful

    Examples:
        >>> data = [{"id": 1}, {"id": 2}]
        >>> save_jsonl(data, "output.jsonl")
        True
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"✅ Saved {len(data)} items to {file_path}")
        return True

    except Exception as e:
        print(f"❌ Error saving {file_path}: {e}")
        return False


def validate_data_structure(
    data: List[Dict],
    required_fields: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate that data has required fields.

    Args:
        data: List of data items
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, errors)

    Examples:
        >>> data = [{"id": 1, "name": "Test"}]
        >>> valid, errors = validate_data_structure(data, ["id", "name"])
        >>> valid
        True
    """
    errors = []

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            errors.append(f"Item {i}: Not a dictionary")
            continue

        for field in required_fields:
            if field not in item:
                errors.append(f"Item {i}: Missing field '{field}'")

    return len(errors) == 0, errors


def filter_data(
    data: List[Dict],
    filter_func,
    keep_none_values: bool = False
) -> List[Dict]:
    """
    Filter data based on predicate function.

    Args:
        data: List of data items
        filter_func: Function that returns True to keep item
        keep_none_values: Whether to keep items with None values

    Returns:
        Filtered list

    Examples:
        >>> data = [{"score": 0.9}, {"score": 0.5}, {"score": 0.7}]
        >>> filtered = filter_data(data, lambda x: x["score"] > 0.6)
        >>> len(filtered)
        2
    """
    filtered = []

    for item in data:
        try:
            keep = filter_func(item)
            if keep or (keep_none_values and keep is None):
                filtered.append(item)
        except Exception as e:
            print(f"⚠️  Error evaluating filter: {e}")

    return filtered


def group_by_field(
    data: List[Dict],
    field: str
) -> Dict[Any, List[Dict]]:
    """
    Group data by field value.

    Args:
        data: List of data items
        field: Field name to group by

    Returns:
        Dictionary mapping field values to lists of items

    Examples:
        >>> data = [{"source": "A", "val": 1}, {"source": "B", "val": 2}]
        >>> groups = group_by_field(data, "source")
        >>> list(groups.keys())
        ['A', 'B']
    """
    groups = {}

    for item in data:
        key = item.get(field)
        if key not in groups:
            groups[key] = []
        groups[key].append(item)

    return groups


def merge_datasets(
    *datasets: List[Dict],
    deduplicate_by: Optional[str] = None
) -> List[Dict]:
    """
    Merge multiple datasets.

    Args:
        *datasets: Variable number of dataset lists
        deduplicate_by: Field name to deduplicate by (None = no dedup)

    Returns:
        Merged dataset

    Examples:
        >>> data1 = [{"id": 1, "name": "A"}]
        >>> data2 = [{"id": 2, "name": "B"}]
        >>> merged = merge_datasets(data1, data2)
        >>> len(merged)
        2
    """
    merged = []

    for dataset in datasets:
        if isinstance(dataset, list):
            merged.extend(dataset)

    if not deduplicate_by:
        return merged

    # Deduplicate
    seen = {}
    result = []

    for item in merged:
        key = item.get(deduplicate_by)
        if key not in seen:
            seen[key] = True
            result.append(item)

    print(f"📊 Merged {len(merged)} items, {len(result)} unique")
    return result


def get_data_statistics(data: List[Dict]) -> Dict[str, Any]:
    """
    Compute statistics about dataset.

    Args:
        data: List of data items

    Returns:
        Dictionary with statistics

    Examples:
        >>> data = [{"score": 0.9}, {"score": 0.8}, {"score": 0.7}]
        >>> stats = get_data_statistics(data)
        >>> stats["num_items"]
        3
    """
    if not data:
        return {"error": "Empty dataset"}

    import statistics

    stats = {
        "num_items": len(data),
        "num_fields": len(set(k for item in data for k in item.keys())),
        "fields": list(set(k for item in data for k in item.keys())),
    }

    # Find numeric fields
    for field in stats["fields"]:
        values = [item.get(field) for item in data if isinstance(item.get(field), (int, float))]
        if values:
            stats[f"{field}_stats"] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
            }

    return stats
