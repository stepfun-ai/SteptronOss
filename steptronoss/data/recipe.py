"""Base data structures for managing dataset configurations.

This module provides the DataList class for organizing data paths and mount points
used in SFT and RL training configurations.
"""

import random
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class DataList:
    """A container for dataset paths and their associated mount points.

    This class manages collections of data file paths along with their required
    mount points for distributed training environments. It supports combining
    multiple DataList instances and provides convenient methods for validation
    and introspection.

    Attributes:
        data_paths: list of absolute paths to dataset files
        mounts: list of mount point specifications (e.g., "juicefs+s3://...")

    Example:
        >>> data_list = DataList(
        ...     data_paths=["/path/to/dataset1.json", "/path/to/dataset2.json"],
        ...     mounts=["juicefs+s3://bucket:/mnt/data"]
        ... )
        >>> print(f"Total files: {len(data_list)}")
        Total files: 2
    """

    data_paths: list[str] = field(default_factory=list)
    mounts: list[str] = field(default_factory=list)

    def __add__(self, other: "DataList") -> "DataList":
        """Combine two DataList instances.

        Args:
            other: Another DataList to combine with this one

        Returns:
            A new DataList containing paths and mounts from both instances
        """
        return DataList(
            data_paths=self.data_paths + other.data_paths,
            mounts=self.mounts + other.mounts,
        )

    def __len__(self) -> int:
        """Return the number of data paths."""
        return len(self.data_paths)

    def __iter__(self) -> Iterator[str]:
        """Iterate over data paths."""
        return iter(self.data_paths)

    def __bool__(self) -> bool:
        """Return True if there are any data paths."""
        return bool(self.data_paths)

    def is_empty(self) -> bool:
        """Check if the DataList contains no data paths.

        Returns:
            True if data_paths is empty, False otherwise
        """
        return len(self.data_paths) == 0

    def unique_mounts(self) -> list[str]:
        """Get unique mount points, preserving order.

        Returns:
            list of unique mount specifications
        """
        seen = set()
        unique = []
        for mount in self.mounts:
            if mount not in seen:
                seen.add(mount)
                unique.append(mount)
        return unique

    def validate_paths(self) -> list[str]:
        """Validate that all data paths are absolute paths.

        Returns:
            list of invalid (non-absolute) paths, empty if all are valid
        """
        invalid_paths = []
        for path in self.data_paths:
            if not path.startswith("/"):
                invalid_paths.append(path)
        return invalid_paths

    def summary(self) -> str:
        """Generate a summary of the DataList contents.

        Returns:
            A formatted string describing the DataList contents
        """
        return f"DataList: {len(self.data_paths)} files, " f"{len(self.unique_mounts())} unique mounts"


# --- Domain data modeling---


@dataclass
class DataSourceFile:
    """
    Represents a single data source file and its subsampling rate.
    """

    path: str
    """Absolute or relative path to the file."""

    subsample_rate: float = 1.0
    """
    The subsampling rate for this specific file (0.0, 1.0].
    Defaults to 1.0, meaning all data in this file is used.
    """

    def __post_init__(self):
        """
        Validates the fields of this specific file.
        """
        if not self.path:
            raise ValueError("DataSourceFile 'path' cannot be empty.")

        if not (0.0 < self.subsample_rate <= 1.0):
            raise ValueError(
                f"DataSourceFile 'subsample_rate' must be in the range (0.0, 1.0]. "
                f"Got: {self.subsample_rate} for file '{self.path}'"
            )


@dataclass
class DomainData:
    """
    Represents a data domain, composed of one or more data source files.
    """

    files: list[DataSourceFile]
    """A list of files belonging to this data domain."""

    def __post_init__(self):
        """
        Validates the domain data.
        """
        if not self.files:
            raise ValueError("DomainData 'files' list cannot be empty.")

    @classmethod
    def from_legacy_data(cls, data_paths: list[str | tuple[str, float]]) -> "DomainData":
        """
        Convert from a legacy data paths list to a DomainData.

        data_paths is a list where each element can be either:
            1. A string (str), representing a json file path (subsample_rate defaults to 1.0).
            2. A tuple (tuple[str, float]), containing a json file path and its subsample rate.
        """
        if not data_paths:
            raise ValueError("Cannot create DomainData from empty data_paths list.")

        data_source_files: list[DataSourceFile] = []

        # Iterate over the mixed list and check the type of each item
        for item in data_paths:
            if isinstance(item, str):
                # Case 1: Item is a string (path only)
                data_source_files.append(DataSourceFile(path=item, subsample_rate=1.0))

            elif isinstance(item, tuple):
                # Case 2: Item is a tuple (path, rate)
                # Perform validation on the tuple structure
                if not (len(item) == 2 and isinstance(item[0], str) and isinstance(item[1], (float, int))):
                    raise TypeError(f"Malformed tuple in data_paths list: {item}. " "Expected (str, float).")

                path, rate = item
                data_source_files.append(DataSourceFile(path=path, subsample_rate=float(rate)))

            else:
                # Handle unexpected types
                raise TypeError(
                    f"Unsupported item type in data_paths: {type(item)}. " "Expected str or tuple[str, float]."
                )

        # The new cls(files=...) instance will automatically trigger
        # the __post_init__ checks for both DomainData and DataSourceFile.
        return cls(files=data_source_files)

    def get_shuffled_raw_files(self):
        # Convert to list[tuple[str, float]], shuffle and then return
        raw_files = [[source_file.path, source_file.subsample_rate] for source_file in self.files]
        random.Random(1234).shuffle(raw_files)
        return raw_files


# --- The data recipe ---


@dataclass
class DataRecipe:
    """
    A complete model for SFT training data and its sampling plan.
    """

    domains: dict[str, DomainData]
    """
    Definition of the data domains.
    Key is the unique name of the domain (e.g., "wiki", "code", "chat").
    Value is the DomainData object for that domain.
    """

    epochs: dict[str, float]
    """
    The sampling plan expressed as epochs per domain.
    Keys must correspond exactly to the keys in 'domains'.
    """

    def __post_init__(self):
        """
        Performs validation across the entire configuration.
        """
        if not self.domains:
            raise ValueError("DataRecipe 'domains' cannot be empty.")
        if not self.epochs:
            raise ValueError("DataRecipe 'epochs' cannot be empty.")

        # Validation 1: Every key in 'epochs' must also be defined in 'domains'.
        for domain_name in self.epochs:
            if domain_name not in self.domains:
                raise ValueError(f"Domain '{domain_name}' from 'epochs' is not defined in 'domains'.")

        # Validation 2: Every key in 'domains' must also have a configuration in 'epochs'.
        if set(self.domains.keys()) != set(self.epochs.keys()):
            missing_in_epochs = set(self.domains.keys()) - set(self.epochs.keys())
            if missing_in_epochs:
                raise ValueError(
                    f"Data domain(s) {missing_in_epochs} are missing a corresponding configuration in 'epochs'."
                )

        # Validation 3: Every epoch must be positive.
        for domain_name, epoch in self.epochs.items():
            if epoch <= 0.0:
                raise ValueError(f"Epochs for domain '{domain_name}' must be positive. Got: {epoch}")

    def to_dict(self, rep=False):
        return {"domains": self.domains, "epochs": self.epochs}

    def __eq__(self, value):
        if isinstance(value, DataRecipe):
            return self.domains == value.domains and self.epochs == value.epochs
        else:
            return False


@dataclass
class CompiledDataRecipe:
    """
    A complete model for SFT training data and its sampling plan.
    """

    domains: dict[str, str]
    """
    Definition of the data domains.
    Key is the unique name of the domain (e.g., "wiki", "code", "chat").
    Value is the compiled path.
    """

    epochs: dict[str, float]
    """
    The sampling plan expressed as epochs per domain.
    Keys must correspond exactly to the keys in 'domains'.
    """

    def __eq__(self, value):
        if isinstance(value, CompiledDataRecipe):
            return self.domains == value.domains and self.epochs == value.epochs
        else:
            return False
