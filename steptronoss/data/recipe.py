"""Base data structures for managing dataset configurations.

This module provides the DataList class for organizing data paths and mount points
used in SFT and RL training configurations.
"""

from dataclasses import dataclass

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


# --- The data recipe ---


@dataclass
class DataRecipe:
    """
    A complete model for SFT training data and its sampling plan.
    """

    domains: dict[str, list[DataSourceFile]]
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
