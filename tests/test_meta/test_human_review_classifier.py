import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from sklego.meta import HumanReviewClassifier


class FixedProbaClassifier(ClassifierMixin, BaseEstimator):
    """Minimal probabilistic classifier that returns canned probabilities and records its fits."""

    def __init__(self, proba, classes=None):
        self.proba = np.asarray(proba, dtype=float)
        self.classes = classes
        self.n_fits = 0

    def fit(self, X, y, sample_weight=None):
        self.n_fits += 1
        self.classes_ = np.asarray(self.classes) if self.classes is not None else np.array([0, 1])
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        assert X.shape[0] == self.proba.shape[0]
        return self.proba

    def predict(self, X):
        classes = np.asarray(self.classes) if self.classes is not None else np.array([0, 1])
        return classes[self.predict_proba(X).argmax(axis=1)]


@parametrize_with_checks([HumanReviewClassifier(LogisticRegression(), review_capacity=2, check_input=True)])
def test_sklearn_compatible_estimator(estimator, check):
    if check.func.__name__ in {
        "check_fit2d_1feature",  # custom error message
        "check_sample_weight_equivalence_on_dense_data",  # sample weights are delegated to the wrapped model
        # `predict` intentionally returns a third `review_label` (object dtype), which breaks assumptions that
        # predictions only contain the target classes and share their dtype.
        "check_classifiers_train",
        "check_classifiers_classes",
        "check_classifiers_regression_target",
        "check_supervised_y_2d",
        "check_estimators_pickle",
        "check_classifier_data_not_an_array",
        "check_methods_sample_order_invariance",
        "check_methods_subset_invariance",
        "check_fit_idempotent",
        "check_fit_score_takes_y",
    }:
        pytest.skip()
    check(estimator)


def test_requires_probabilistic_model():
    mod = HumanReviewClassifier(LinearRegression())
    X = np.random.normal(0, 1, (20, 2))
    y = np.array([0] * 10 + [1] * 10)
    with pytest.raises(ValueError):
        mod.fit(X, y)


def test_requires_binary_target():
    mod = HumanReviewClassifier(LogisticRegression())
    X = np.random.normal(0, 1, (30, 2))
    y = np.repeat(["a", "b", "c"], 10)
    with pytest.raises(ValueError):
        mod.fit(X, y)


def test_invalid_costs_and_capacity():
    X = np.random.normal(0, 1, (20, 2))
    y = np.array([0] * 10 + [1] * 10)
    with pytest.raises(ValueError):
        HumanReviewClassifier(LogisticRegression(), false_positive_cost=-1).fit(X, y)
    with pytest.raises(ValueError):
        HumanReviewClassifier(LogisticRegression(), review_cost=np.nan).fit(X, y)
    with pytest.raises(ValueError):
        HumanReviewClassifier(LogisticRegression(), review_capacity=1.5).fit(X, y)
    with pytest.raises(ValueError):
        HumanReviewClassifier(LogisticRegression(), review_capacity=-1).fit(X, y)


def test_review_label_must_differ_from_classes():
    X = np.random.normal(0, 1, (20, 2))
    y = np.array(["a"] * 10 + ["b"] * 10)
    with pytest.raises(ValueError):
        HumanReviewClassifier(LogisticRegression(), review_label="a").fit(X, y)


def test_each_fit_trains_independent_clone():
    inner = FixedProbaClassifier(np.column_stack([np.linspace(0.9, 0.1, 6), np.linspace(0.1, 0.9, 6)]))
    mod = HumanReviewClassifier(inner, review_cost=0.3, review_capacity=2, refit=True)
    X = np.zeros((6, 2))
    y = np.array([0, 0, 0, 1, 1, 1])
    mod.fit(X, y)
    first = mod.estimator_
    assert first is not inner  # a clone was trained, not the prototype
    assert first.n_fits == 1
    mod.fit(X, y)
    assert mod.estimator_ is not first  # each fit produces a fresh independent estimator
    assert mod.estimator_.n_fits == 1


def test_refit_false_reuses_fitted_model():
    inner = FixedProbaClassifier(np.column_stack([np.linspace(0.9, 0.1, 6), np.linspace(0.1, 0.9, 6)]))
    inner.fit(np.zeros((6, 2)), np.zeros(6))
    mod = HumanReviewClassifier(inner, review_cost=0.3, review_capacity=2, refit=False)
    mod.fit(np.zeros((6, 2)), np.array([0, 0, 0, 1, 1, 1]))
    assert mod.estimator_ is inner
    assert inner.n_fits == 1


def test_basic_review_selection_and_capacity():
    # probabilities of the positive class
    p = np.array([0.05, 0.45, 0.55, 0.45, 0.95])
    proba = np.column_stack([1 - p, p])
    mod = HumanReviewClassifier(
        FixedProbaClassifier(proba),
        false_positive_cost=2.0,
        false_negative_cost=4.0,
        review_cost=0.8,
        review_capacity=2,
    )
    X = np.zeros((5, 2))
    y = np.array([0, 0, 1, 1, 1])
    mod.fit(X, y)

    # expected automatic losses
    pos_loss = (1 - p) * 2.0  # cost of saying positive
    neg_loss = p * 4.0  # cost of saying negative
    auto = np.where(pos_loss < neg_loss, 1, 0)
    benefit = np.minimum(pos_loss, neg_loss) - 0.8
    # pos_loss [1.9, 1.1, 0.9, 1.1, 0.1], neg_loss [0.2, 1.8, 2.2, 1.8, 3.8]
    np.testing.assert_allclose(benefit, [-0.6, 0.3, 0.1, 0.3, -0.7])

    preds = mod.predict(X)
    # top two benefits are tied at 0.3 (idx 1 and 3); stable row order gives both slots, capacity is exactly 2
    assert preds[[1, 3]].tolist() == ["review", "review"]
    assert (preds == "review").sum() == 2
    # the remaining rows keep their cost-minimizing automatic decision
    assert preds[[0, 2, 4]].tolist() == [auto[0], auto[2], auto[4]]
    assert auto.tolist() == [0, 1, 1, 1, 1]


def test_zero_benefit_does_not_consume_capacity():
    p = np.array([0.5, 0.5])
    proba = np.column_stack([1 - p, p])
    # automatic loss = 0.5 * 2 = 1.0; review cost 1.0 -> zero benefit everywhere
    mod = HumanReviewClassifier(
        FixedProbaClassifier(proba), false_positive_cost=2, false_negative_cost=2, review_cost=1.0, review_capacity=5
    )
    mod.fit(np.zeros((2, 2)), np.array([0, 1]))
    preds = mod.predict(np.zeros((2, 2)))
    assert "review" not in preds.tolist()


def test_negative_benefit_never_reviewed():
    p = np.array([0.99, 0.01])
    proba = np.column_stack([1 - p, p])
    mod = HumanReviewClassifier(
        FixedProbaClassifier(proba), false_positive_cost=1, false_negative_cost=1, review_cost=0.5, review_capacity=5
    )
    mod.fit(np.zeros((2, 2)), np.array([0, 1]))
    preds = mod.predict(np.zeros((2, 2)))
    assert "review" not in preds.tolist()
    # confident rows follow the argmax decision
    assert preds.tolist() == [1, 0]


def test_tie_in_automatic_decision_picks_negative_class():
    # p * c_fn == (1 - p) * c_fp with equal costs happens at p = 0.5
    proba = np.array([[0.5, 0.5]])
    mod = HumanReviewClassifier(
        FixedProbaClassifier(proba, classes=np.array(["neg", "pos"])),
        false_positive_cost=3,
        false_negative_cost=3,
        review_cost=10.0,  # review never worthwhile
        review_capacity=1,
    )
    mod.fit(np.zeros((1, 2)), np.array(["neg"]))
    preds = mod.predict(np.zeros((1, 2)))
    assert preds[0] == "neg"


def test_capacity_zero_behaves_like_cost_minimizing_classifier():
    p = np.array([0.1, 0.5, 0.9])
    proba = np.column_stack([1 - p, p])
    mod = HumanReviewClassifier(
        FixedProbaClassifier(proba), false_positive_cost=2, false_negative_cost=4, review_cost=0.1, review_capacity=0
    )
    mod.fit(np.zeros((3, 2)), np.array([0, 0, 1]))
    preds = mod.predict(np.zeros((3, 2)))
    pos_loss = (1 - p) * 2
    neg_loss = p * 4
    assert preds.tolist() == np.where(pos_loss < neg_loss, 1, 0).tolist()


def test_string_and_discontinuous_integer_classes_keep_meaning():
    p = np.array([0.2, 0.8])
    proba = np.column_stack([1 - p, p])
    mod = HumanReviewClassifier(
        FixedProbaClassifier(proba, classes=np.array(["cat", "dog"])),
        false_positive_cost=1,
        false_negative_cost=1,
        review_cost=0.95,  # automatic losses are 0.2 at most, so review is never worthwhile
        review_capacity=1,
    )
    mod.fit(np.zeros((2, 2)), np.array(["cat", "dog"]))
    preds = mod.predict(np.zeros((2, 2)))
    # benefits: 0.2 - 0.95 < 0 both, so no review; labels come from the real classes
    assert preds.tolist() == ["cat", "dog"]
    assert mod.classes_.tolist() == ["cat", "dog"]

    proba2 = np.column_stack([1 - p, p])
    mod2 = HumanReviewClassifier(
        FixedProbaClassifier(proba2, classes=np.array([7, -3])),
        false_positive_cost=10,
        false_negative_cost=10,
        review_cost=1.0,
        review_capacity=1,
    )
    mod2.fit(np.zeros((2, 2)), np.array([7, -3]))
    # idx 0 (p=0.2): losses pos=8, neg=2 -> auto negative (7); benefit 8 - 1 = 7 -> reviewed
    # idx 1 (p=0.8): losses pos=2, neg=8 -> auto positive (-3); benefit 2 - 1 = 1
    preds2 = mod2.predict(np.zeros((2, 2)))
    assert preds2.tolist() == ["review", -3]
    assert mod2.classes_.tolist() == [7, -3]


def test_predict_proba_has_only_two_real_class_columns():
    p = np.array([0.3, 0.7])
    proba = np.column_stack([1 - p, p])
    mod = HumanReviewClassifier(
        FixedProbaClassifier(proba, classes=np.array(["a", "b"])), review_cost=0.01, review_capacity=2
    )
    mod.fit(np.zeros((2, 2)), np.array(["a", "b"]))
    out = mod.predict_proba(np.zeros((2, 2)))
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out, proba)
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_score_reflects_returned_labels_not_underlying_score():
    p = np.array([0.5, 0.9])
    proba = np.column_stack([1 - p, p])
    inner = FixedProbaClassifier(proba)
    mod = HumanReviewClassifier(inner, false_positive_cost=2, false_negative_cost=2, review_cost=0.5, review_capacity=1)
    X = np.zeros((2, 2))
    # y matches the underlying argmax predictions, so the underlying accuracy is 1.0
    y = np.array([0, 1])
    mod.fit(X, y)
    # idx 0: automatic loss 1.0, benefit 0.5 -> reviewed; idx 1 auto predicts 1 and is correct
    assert mod.predict(X).tolist() == ["review", 1]
    # review row never equals a true class label -> accuracy 0.5, not the underlying argmax accuracy of 1.0
    assert mod.score(X, y) == 0.5
    argmax_preds = proba.argmax(axis=1)
    assert float(np.mean(argmax_preds == y)) == 1.0
    assert mod.score(X, y) != float(np.mean(argmax_preds == y))


def test_single_row_prediction():
    proba = np.array([[0.5, 0.5]])
    mod = HumanReviewClassifier(
        FixedProbaClassifier(proba), false_positive_cost=2, false_negative_cost=2, review_cost=0.5, review_capacity=1
    )
    mod.fit(np.zeros((1, 2)), np.array([0]))
    preds = mod.predict(np.zeros((1, 2)))
    assert preds.shape == (1,)
    assert preds[0] == "review"  # benefit 1.0 - 0.5 > 0


def test_preserves_pandas_index_shuffled_and_duplicated():
    p = np.array([0.5, 0.5, 0.5, 0.99])
    proba = np.column_stack([1 - p, p])
    inner = FixedProbaClassifier(proba)
    mod = HumanReviewClassifier(inner, false_positive_cost=2, false_negative_cost=2, review_cost=0.5, review_capacity=2)
    idx = pd.Index([3, 0, 3, 1])  # shuffled and duplicated
    X = pd.DataFrame(np.zeros((4, 2)), index=idx)
    y = pd.Series([0, 1, 0, 1], index=idx)
    mod.fit(X, y)
    preds = mod.predict(X)
    assert isinstance(preds, pd.Series)
    assert preds.index.tolist() == idx.tolist()
    # benefits for the three 0.5 rows are 0.5 each; capacity 2, stable order -> rows at positional idx 0, 1
    assert preds.iloc[[0, 1]].tolist() == ["review", "review"]
    assert preds.iloc[2] == 0  # tie, missed capacity, automatic tie -> negative class
    assert preds.iloc[3] == 1  # confident positive, no review benefit


def test_pipeline_support():
    np.random.seed(42)
    X = np.random.normal(0, 1, (60, 3))
    y = (X[:, 0] + np.random.normal(0, 0.5, 60)) > 0
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("review", HumanReviewClassifier(LogisticRegression(), review_cost=0.2, review_capacity=3)),
        ]
    )
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == (60,)
    assert (preds == "review").sum() <= 3
    assert pipe.predict_proba(X).shape == (60, 2)
    assert 0.0 <= pipe.score(X, y) <= 1.0


def test_repeated_fit_predict_cycles_are_stable():
    np.random.seed(0)
    mod = HumanReviewClassifier(LogisticRegression(), review_cost=0.3, review_capacity=2)
    X = np.random.normal(0, 1, (40, 2))
    y = (X[:, 0] > 0).astype(int)
    mod.fit(X, y)
    first = mod.predict(X).copy()
    mod.fit(X, y)
    second = mod.predict(X)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))


def test_sample_weight_passed_through():
    class WeightedLR(LogisticRegression):
        def fit(self, X, y, sample_weight=None):
            assert sample_weight is not None
            return super().fit(X, y, sample_weight=sample_weight)

    mod = HumanReviewClassifier(WeightedLR(), review_capacity=1)
    X = np.random.normal(0, 1, (30, 2))
    y = (X[:, 0] > 0).astype(int)
    mod.fit(X, y, sample_weight=np.ones(30))


def test_dummy_classifier_end_to_end():
    X = np.arange(20).reshape(-1, 1)
    y = np.array([0] * 12 + [1] * 8)
    mod = HumanReviewClassifier(
        DummyClassifier(strategy="uniform", random_state=0),
        false_positive_cost=3,
        false_negative_cost=5,
        review_cost=0.4,
        review_capacity=4,
    )
    mod.fit(X, y)
    preds = mod.predict(X).tolist()
    assert set(preds).issubset({0, 1, "review"})
    assert sum(label == "review" for label in preds) <= 4
