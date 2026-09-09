from __future__ import annotations

import numpy as np


def leaf_cooccurrence_match(
    donor_leaves: np.ndarray,
    recipient_leaves: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For each recipient row, pick a donor uniformly at random from the
    multiset of donors that share a leaf with it in ANY column (tree) of
    donor_leaves/recipient_leaves - donors sharing leaves across more
    trees are proportionally more likely to be picked, exactly like
    mice's mice.impute.rf: pool candidate donors from every tree's
    matching terminal node, then draw one uniformly from the pool.

    This never materializes that pool. It streams column-by-column
    (tree-by-tree) and maintains a running weighted reservoir sample of
    size 1 per recipient (Chao's algorithm, generalized to batched
    insertion - see the module docstring below for the derivation). Memory
    is O(n_recipients), independent of n_trees or how large the true
    candidate pools would be - nothing scales with total match count.

    Per tree t:
        1. Sort donor_leaves[:, t] (a groupby-via-sort) so every
           recipient's matching donors become one contiguous slice,
           found by two searchsorted calls (vectorized over all
           recipients at once).
        2. That slice ("batch") has some size k (0 if no match). If
           k > 0, replace the recipient's currently-held donor with a
           uniformly random member of the batch, with probability
           k / (count_so_far + k) - the standard weighted-reservoir
           replacement rule, applied to a batch instead of a single item.
        3. count_so_far += k.

    After all trees, current_donor holds one donor row index per
    recipient (or -1 if no donor ever shared a leaf with it in any tree).

    This makes no assumption about which library produced the leaf
    matrices, or how many trees there are - only that each column is some
    integer leaf/node id, comparable within that column. It is exactly as
    valid for RandomForest's one-leaf-per-tree columns as for a boosting
    model's multiple leaf-columns per round (e.g. one per class, before
    a shared/vector-leaf tree structure is used) - see
    Parameters._categorical_enum_dtypes/RandomForest()/XGBoost()/
    CatBoost() for the analogous "framework stays agnostic, each model
    owns its own quirks" pattern used elsewhere for tabular-ML imputation.

    Parameters
    ----------
    donor_leaves : np.ndarray
        Shape (n_donors, n_trees) - donor_leaves[i, t] is donor i's leaf/
        node id in tree t. Any integer dtype.
    recipient_leaves : np.ndarray
        Shape (n_recipients, n_trees) - same leaf/node id space as
        donor_leaves (i.e. produced by the same fitted model), for the
        rows needing a donor.
    rng : np.random.Generator
        Source of randomness - pass survey_kit.utilities.random's own
        RandomNumberGenerator() for the usual seeded-reproducibility
        convention used elsewhere in this codebase.

    Returns
    -------
    np.ndarray
        Shape (n_recipients,) int64 - for each recipient, the row index
        into donor_leaves of the donor selected for it, or -1 if that
        recipient shared no leaf with any donor in any tree (an
        essentially-degenerate case for a real forest/booster with a
        reasonable donor pool, but possible with pathologically small
        leaves or a tiny donor pool - callers should check for -1 rather
        than assume every recipient gets matched).
    """
    n_donors, n_trees = donor_leaves.shape
    n_recipients, n_trees_recipient = recipient_leaves.shape
    if n_trees != n_trees_recipient:
        raise ValueError(
            f"donor_leaves has {n_trees} tree columns but recipient_leaves "
            f"has {n_trees_recipient} - both must come from the same "
            f"fitted model's leaf-index output."
        )

    current_donor = np.full(n_recipients, -1, dtype=np.int64)
    count_so_far = np.zeros(n_recipients, dtype=np.int64)

    if n_donors == 0:
        return current_donor

    for t in range(n_trees):
        donor_col = donor_leaves[:, t]
        #   argsort, not a dict/groupby - a sorted array plus two
        #       searchsorted calls below finds every recipient's matching
        #       donor slice in one vectorized pass, without ever
        #       building a hash table or a (recipient, donor) pair list.
        order = np.argsort(donor_col, kind="stable")
        sorted_vals = donor_col[order]

        recip_col = recipient_leaves[:, t]
        lo = np.searchsorted(sorted_vals, recip_col, side="left")
        hi = np.searchsorted(sorted_vals, recip_col, side="right")
        bucket_size = hi - lo

        has_match = bucket_size > 0
        if not np.any(has_match):
            continue

        new_count = count_so_far + bucket_size

        #   A uniformly random representative from THIS tree's matching
        #       slice, for every recipient (computed for all recipients,
        #       not just matched ones - np.where below discards it for
        #       recipients with no match this tree, but np.where always
        #       evaluates both branches, so the indexing has to stay in
        #       bounds for everyone).
        offsets = np.zeros(n_recipients, dtype=np.int64)
        offsets[has_match] = np.minimum(
            (rng.random(int(has_match.sum())) * bucket_size[has_match]).astype(np.int64),
            bucket_size[has_match] - 1,
        )
        candidate_sorted_pos = np.clip(lo + offsets, 0, n_donors - 1)
        candidate_donor_idx = np.where(
            has_match, order[candidate_sorted_pos], -1
        )

        #   Weighted-reservoir replacement: this tree's batch (size k)
        #       replaces the running pick with probability
        #       k / (count_so_far + k) - see the module docstring's
        #       derivation. count_so_far == 0 (first-ever match) always
        #       replaces, since k / (0 + k) == 1.
        replace_prob = np.zeros(n_recipients)
        replace_prob[has_match] = (
            bucket_size[has_match] / new_count[has_match]
        )
        do_replace = has_match & (rng.random(n_recipients) < replace_prob)

        current_donor[do_replace] = candidate_donor_idx[do_replace]
        count_so_far = new_count

    return current_donor


def extract_leaf_indices(model: object, X: object) -> np.ndarray:
    """
    Per-tree leaf/node id for each row of X, shape (n_rows, n_trees) -
    the format leaf_cooccurrence_match needs. Different tree-ensemble
    libraries expose this under different method names, so this just
    tries each in turn:
      - .apply(X) - scikit-learn's RandomForestRegressor/
        RandomForestClassifier and xgboost's XGBRegressor/XGBClassifier
        (sklearn API) both use this name.
      - .calc_leaf_indexes(X) - CatBoostRegressor/CatBoostClassifier's
        own name for the same thing.

    Raises AttributeError (with a message naming what IS supported) for
    any estimator exposing neither - e.g. a plain LinearRegression,
    which has no tree structure to extract at all.
    """
    if hasattr(model, "apply"):
        leaves = model.apply(X)
    elif hasattr(model, "calc_leaf_indexes"):
        leaves = model.calc_leaf_indexes(X)
    else:
        message = (
            f"Leaf-based donor matching needs an estimator that exposes "
            f"per-tree leaf indices (.apply() - scikit-learn/XGBoost - or "
            f".calc_leaf_indexes() - CatBoost), but {type(model).__name__} "
            f"exposes neither."
        )
        raise AttributeError(message)

    return np.asarray(leaves).astype(np.int64)
