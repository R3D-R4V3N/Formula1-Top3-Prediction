import numpy as np
from sklearn.utils.validation import indexable

class GroupTimeSeriesSplit:
    """Time series cross-validator with non-overlapping groups.

    Splits data so that each test fold contains complete groups and
    training indices are all groups that come before it chronologically.
    """

    def __init__(self, n_splits=5):
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("'groups' must be provided")
        (groups,) = indexable(groups)
        unique_groups, first_idx = np.unique(groups, return_index=True)
        order = np.argsort(first_idx)
        groups_ordered = unique_groups[order]
        n_groups = len(groups_ordered)
        n_folds = self.n_splits + 1
        if n_folds > n_groups:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater than number of groups={n_groups}."
            )
        test_size = n_groups // n_folds
        group_to_indices = {g: np.flatnonzero(groups == g) for g in groups_ordered}
        for start in range(n_groups - self.n_splits * test_size, n_groups, test_size):
            train_groups = groups_ordered[:start]
            test_groups = groups_ordered[start:start + test_size]
            train_idx = np.concatenate([group_to_indices[g] for g in train_groups])
            test_idx = np.concatenate([group_to_indices[g] for g in test_groups])
            yield train_idx, test_idx
