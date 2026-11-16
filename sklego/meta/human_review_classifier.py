import logging
from inspect import signature

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn_compat.utils.multiclass import type_of_target
from sklearn_compat.utils.validation import _check_n_features, validate_data

from sklego.base import ProbabilisticClassifier


class HumanReviewClassifier(ClassifierMixin, MetaEstimatorMixin, BaseEstimator):
    """Wraps a binary probabilistic classifier and adds a human-in-the-loop review step with a limited capacity.

    For every batch of predictions the estimator compares the expected loss of an automatic decision with the
    expected cost of sending the row to a human reviewer. Rows are reviewed, up to a maximum of `review_capacity`
    rows per batch, whenever doing so strictly reduces the expected loss. Rows that are not reviewed receive their
    automatically decided class.

    The two target classes keep their real meaning:

    - `classes_[0]` is treated as the negative class,
    - `classes_[1]` is treated as the positive class.

    The costs are expressed in the business' own units:

    - `false_positive_cost` is paid when a negative row is automatically classified as positive (a *misjudgement*),
    - `false_negative_cost` is paid when a positive row is automatically classified as negative (a *missed case*),
    - `review_cost` is paid for every row that is sent to human review.

    Given a positive-class probability `p`, the expected loss of predicting the positive class is
    `(1 - p) * false_positive_cost` and the expected loss of predicting the negative class is
    `p * false_negative_cost`. The automatic decision picks the smaller expected loss and breaks ties in favour of the
    negative class. Reviewing a row has a constant expected loss of `review_cost`; it is worthwhile only when its
    expected benefit, the difference between the automatic loss and the review cost, is strictly positive. Rows are
    ranked by this benefit (ties keep the original row order) and only the best `review_capacity` rows occupy review
    slots, so rows with zero benefit never consume capacity.

    !!! info
        This only works for binary classification problems. The underlying estimator must implement
        `predict_proba`. Probabilities always describe the two real classes; no extra probability column is created
        for the review outcome.

    !!! warning
        `predict` returns three distinct labels: the two target classes plus `review_label`, stored in an
        object-dtype array. sklearn's built-in classification scorers (for example the `"accuracy"` string in
        `GridSearchCV`) cannot mix these dtypes when the target classes are integers or booleans. Use
        `sklearn.metrics.make_scorer` with an object-dtype comparison, or call the estimator's own `score` method,
        which already evaluates the labels actually returned by `predict`.

    Parameters
    ----------
    model : scikit-learn compatible classifier
        Classifier that will be wrapped. It must implement a `predict_proba` method.
    false_positive_cost : float, default=1.0
        Non-negative cost of automatically predicting the positive class for a truly negative row.
    false_negative_cost : float, default=1.0
        Non-negative cost of automatically predicting the negative class for a truly positive row.
    review_cost : float, default=0.5
        Non-negative cost of sending a single row to human review.
    review_capacity : int, default=1
        Maximum number of rows (`B`) that may be reviewed in a single `predict` batch. Must be a non-negative integer.
    review_label : object, default="review"
        Label returned by `predict` for rows that are sent to human review. It must not coincide with either of the
        two target classes.
    refit : bool, default=True
        - If True, every call to `fit` trains a fresh clone of `model`.
        - If False, `model` is only (re)trained when it is not fitted yet.
    check_input : bool, default=False
        Whether or not to check the input data. If False, the checks are delegated to the wrapped estimator.

    Attributes
    ----------
    estimator_ : scikit-learn compatible classifier
        The fitted classifier.
    classes_ : numpy.ndarray of shape (2,)
        The class labels, with the negative class at index 0 and the positive class at index 1.

    Example
    -------

    ```py
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklego.meta import HumanReviewClassifier

    np.random.seed(0)
    n1, n2 = 50, 100
    X = np.concatenate([np.random.normal(0, 1, (n1, 2)), np.random.normal(2, 1, (n2, 2))], axis=0)
    y = np.concatenate([np.zeros((n1, 1)), np.ones((n2, 1))], axis=0).reshape(-1)

    reviewer = HumanReviewClassifier(
        LogisticRegression(),
        false_positive_cost=2.0,
        false_negative_cost=4.0,
        review_cost=0.5,
        review_capacity=5,
    )
    reviewer.fit(X, y)
    preds = reviewer.predict(X)
    # up to five uncertain rows are flagged for review instead of being classified automatically
    ```
    """

    _required_parameters = ["model"]

    def __init__(
        self,
        model,
        false_positive_cost: float = 1.0,
        false_negative_cost: float = 1.0,
        review_cost: float = 0.5,
        review_capacity: int = 1,
        review_label="review",
        refit: bool = True,
        check_input: bool = False,
    ):
        self.model = model
        self.false_positive_cost = false_positive_cost
        self.false_negative_cost = false_negative_cost
        self.review_cost = review_cost
        self.review_capacity = review_capacity
        self.review_label = review_label
        self.refit = refit
        self.check_input = check_input

    def _handle_unfitted(self, X, y, sample_weight):
        sample_weight_ = _check_sample_weight(sample_weight, X)

        self.estimator_ = clone(self.model)
        if "sample_weight" in signature(self.estimator_.fit).parameters:
            self.estimator_.fit(X, y, sample_weight=sample_weight_)
        else:
            if sample_weight is not None:
                logging.warning("Estimator ignores sample_weight.")
            self.estimator_.fit(X, y)
        return self

    def _handle_refit(self, X, y, sample_weight=None):
        """Only refit when we need to, unless `refit=True` is present."""
        if self.refit:
            self._handle_unfitted(X, y, sample_weight)
        else:
            try:
                check_is_fitted(self.estimator_)
            except NotFittedError:
                self._handle_unfitted(X, y, sample_weight)

    def _check_costs(self):
        costs = {
            "false_positive_cost": self.false_positive_cost,
            "false_negative_cost": self.false_negative_cost,
            "review_cost": self.review_cost,
        }
        for name, value in costs.items():
            if not np.isscalar(value) or isinstance(value, (bool, str, bytes)) or not np.isfinite(value):
                raise ValueError(f"`{name}` must be a finite non-negative number, got {value!r}.")
            if value < 0:
                raise ValueError(f"`{name}` must be non-negative, got {value!r}.")

        capacity = self.review_capacity
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)):
            raise ValueError(f"`review_capacity` must be a non-negative integer, got {capacity!r}.")
        if capacity < 0:
            raise ValueError(f"`review_capacity` must be non-negative, got {capacity!r}.")

    def fit(self, X, y, sample_weight=None):
        """Fit the underlying estimator using `X` and `y` as training data.

        When `refit=True` (the default) every call trains a fresh independent clone of `model`. When `refit=False`
        the provided model is only trained when it is not fitted yet.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The binary target values. String labels and non-consecutive integer labels are supported.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : HumanReviewClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            - If `model` is not a classifier or it does not implement `predict_proba`.
            - If the target is not binary.
            - If any cost is negative or non-numeric, or `review_capacity` is not a non-negative integer.
            - If `review_label` coincides with one of the target classes.
        """
        self._check_costs()

        self.estimator_ = self.model
        if not isinstance(self.estimator_, ProbabilisticClassifier):
            raise ValueError(
                "The HumanReviewClassifier meta model only works on classification models with .predict_proba."
            )

        if self.check_input:
            X, y = validate_data(self, X=X, y=y, ensure_all_finite=False, ensure_min_features=0, reset=True)
        else:
            _check_n_features(self, X, reset=True)

        self._handle_refit(X, y, sample_weight)

        self.classes_ = np.asarray(self.estimator_.classes_)
        y_type = type_of_target(y, input_name="y", raise_unknown=True)
        if y_type != "binary" or len(self.classes_) != 2:
            raise ValueError(f"Only binary classification is supported. The type of the target is {y_type}.")

        if self.review_label in set(self.classes_.tolist()):
            raise ValueError(
                f"`review_label` {self.review_label!r} must be different from the target classes "
                f"{self.classes_.tolist()}."
            )

        return self

    def _check_X(self, X, reset):
        if self.check_input:
            return validate_data(self, X=X, ensure_min_features=0, ensure_all_finite=False, reset=reset)
        _check_n_features(self, X, reset=reset)
        return X

    def _decide(self, X):
        """Return the automatic labels and the per-row benefit of sending each row to review."""
        proba = self.estimator_.predict_proba(X)
        p_positive = proba[:, 1]

        positive_loss = (1.0 - p_positive) * self.false_positive_cost
        negative_loss = p_positive * self.false_negative_cost

        # Strict inequality picks the positive class; equal losses therefore fall back to the negative class.
        automatic = np.where(positive_loss < negative_loss, self.classes_[1], self.classes_[0])
        automatic_loss = np.minimum(positive_loss, negative_loss)
        benefit = automatic_loss - self.review_cost
        return automatic, benefit

    def predict(self, X):
        """Predict class labels for `X`, sending at most `review_capacity` rows to human review.

        Rows are ranked by the expected benefit of reviewing them; only strictly positive benefits consume review
        slots and equal benefits are broken by the original row order. Rows outside the capacity receive the
        automatic decision. When the two automatic decisions have the same expected loss the negative class is
        returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to predict. A single row is allowed. When a pandas DataFrame is passed, a pandas Series with the
            same (possibly duplicated or shuffled) index is returned.

        Returns
        -------
        numpy.ndarray or pandas.Series of shape (n_samples,)
            The predicted labels, with `review_label` used for rows that are sent to human review.
        """
        check_is_fitted(self, ["classes_", "estimator_"])
        X_checked = self._check_X(X, reset=False)

        automatic, benefit = self._decide(X_checked)
        n_samples = benefit.shape[0]

        # Stable sort keeps the original row order for equal benefits; zero/negative benefits never take a slot.
        order = np.argsort(-benefit, kind="stable")
        eligible = order[benefit[order] > 0]
        review_idx = eligible[: min(self.review_capacity, n_samples)]

        # Object dtype preserves the original class labels (and their type) alongside the review label.
        labels = automatic.astype(object, copy=True)
        labels[review_idx] = self.review_label

        index = getattr(X, "index", None)
        if index is not None and hasattr(index, "tolist"):
            return pd.Series(labels, index=index, name="prediction")
        return labels

    def predict_proba(self, X):
        """Return the class probabilities of the underlying estimator.

        The two columns describe the two real classes (`classes_[0]` and `classes_[1]`); no review column is added.
        """
        check_is_fitted(self, ["classes_", "estimator_"])
        X_checked = self._check_X(X, reset=False)
        return self.estimator_.predict_proba(X_checked)

    def score(self, X, y):
        """Return the accuracy of the labels actually returned by `predict`, including review rows.

        Rows flagged for review never equal a target class (the review label is validated to differ from both
        classes), so they count as incorrect. This deliberately differs from the underlying estimator's score,
        which does not account for the review decisions.
        """
        check_is_fitted(self, ["classes_", "estimator_"])
        preds = np.asarray(self.predict(X), dtype=object)
        return float(np.mean(preds == np.asarray(y, dtype=object)))

    def _more_tags(self):
        return {
            "binary_only": True,
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
