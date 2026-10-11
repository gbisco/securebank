import pytest
import pandas as pd

from securebank.modules.train.splitter import DataSplitter


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "trans_num": [f"t{i}" for i in range(20)],
        "unix_time": list(range(1000, 1020)),
        "amt": [float(i * 10) for i in range(20)],
        "is_fraud": [0, 1] * 10,
    })


def test_constructor_defaults_to_holdout():
    splitter = DataSplitter()

    assert splitter.strategy == "holdout"
    assert splitter.validation_size == 0.1
    assert splitter.test_size == 0.2
    assert splitter.random_state == 42
    assert splitter.stratify_column == "is_fraud"
    assert splitter.time_column == "unix_time"


def test_constructor_accepts_time_based_strategy():
    splitter = DataSplitter(strategy="time_based")

    assert splitter.strategy == "time_based"


def test_constructor_rejects_invalid_strategy():
    with pytest.raises(ValueError, match="Split strategy not supported"):
        DataSplitter(strategy="bad_strategy")


def test_constructor_rejects_invalid_validation_size_below_zero():
    with pytest.raises(ValueError, match="validation_size"):
        DataSplitter(validation_size=-0.1)


def test_constructor_rejects_invalid_validation_size_equal_one():
    with pytest.raises(ValueError, match="validation_size"):
        DataSplitter(validation_size=1.0)


def test_constructor_rejects_invalid_test_size_zero():
    with pytest.raises(ValueError, match="test_size"):
        DataSplitter(test_size=0)


def test_constructor_rejects_invalid_test_size_equal_one():
    with pytest.raises(ValueError, match="test_size"):
        DataSplitter(test_size=1.0)


def test_constructor_rejects_validation_plus_test_too_large():
    with pytest.raises(ValueError, match="validation_size \\+ test_size"):
        DataSplitter(validation_size=0.5, test_size=0.5)


def test_split_rejects_empty_dataframe():
    splitter = DataSplitter()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Cannot split an empty DataFrame"):
        splitter.split(empty_df)


def test_holdout_split_returns_train_validation_and_test(sample_df):
    splitter = DataSplitter(
        strategy="holdout",
        validation_size=0.1,
        test_size=0.2,
        random_state=42,
        stratify_column="is_fraud",
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(validation_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)

    assert len(train_df) == 14
    assert len(validation_df) == 2
    assert len(test_df) == 4

    assert len(train_df) + len(validation_df) + len(test_df) == len(sample_df)


def test_holdout_split_resets_indexes(sample_df):
    splitter = DataSplitter(
        strategy="holdout",
        validation_size=0.1,
        test_size=0.2,
        random_state=42,
        stratify_column="is_fraud",
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert train_df.index.tolist() == list(range(len(train_df)))
    assert validation_df.index.tolist() == list(range(len(validation_df)))
    assert test_df.index.tolist() == list(range(len(test_df)))


def test_holdout_split_with_zero_validation_size(sample_df):
    splitter = DataSplitter(
        strategy="holdout",
        validation_size=0,
        test_size=0.2,
        random_state=42,
        stratify_column="is_fraud",
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert len(train_df) == 16
    assert validation_df.empty
    assert len(test_df) == 4

    assert validation_df.columns.tolist() == sample_df.columns.tolist()


def test_holdout_split_preserves_stratified_balance(sample_df):
    splitter = DataSplitter(
        strategy="holdout",
        validation_size=0.1,
        test_size=0.2,
        random_state=42,
        stratify_column="is_fraud",
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert train_df["is_fraud"].value_counts().to_dict() == {0: 7, 1: 7}
    assert validation_df["is_fraud"].value_counts().to_dict() == {0: 1, 1: 1}
    assert test_df["is_fraud"].value_counts().to_dict() == {0: 2, 1: 2}


def test_holdout_split_can_disable_stratification(sample_df):
    splitter = DataSplitter(
        strategy="holdout",
        validation_size=0.1,
        test_size=0.2,
        random_state=42,
        stratify_column=None,
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert len(train_df) == 14
    assert len(validation_df) == 2
    assert len(test_df) == 4


def test_holdout_split_rejects_missing_stratify_column(sample_df):
    splitter = DataSplitter(
        strategy="holdout",
        validation_size=0.1,
        test_size=0.2,
        random_state=42,
        stratify_column="missing_column",
    )

    with pytest.raises(ValueError, match="Stratify column not found"):
        splitter.split(sample_df)


def test_time_based_split_returns_train_validation_and_test(sample_df):
    splitter = DataSplitter(
        strategy="time_based",
        validation_size=0.1,
        test_size=0.2,
        time_column="unix_time",
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert len(train_df) == 14
    assert len(validation_df) == 2
    assert len(test_df) == 4

    assert len(train_df) + len(validation_df) + len(test_df) == len(sample_df)


def test_time_based_split_uses_time_order(sample_df):
    shuffled_df = sample_df.sample(frac=1, random_state=99).reset_index(drop=True)

    splitter = DataSplitter(
        strategy="time_based",
        validation_size=0.1,
        test_size=0.2,
        time_column="unix_time",
    )

    train_df, validation_df, test_df = splitter.split(shuffled_df)

    assert train_df["unix_time"].max() < validation_df["unix_time"].min()
    assert validation_df["unix_time"].max() < test_df["unix_time"].min()

    assert train_df["unix_time"].tolist() == list(range(1000, 1014))
    assert validation_df["unix_time"].tolist() == list(range(1014, 1016))
    assert test_df["unix_time"].tolist() == list(range(1016, 1020))


def test_time_based_split_resets_indexes(sample_df):
    splitter = DataSplitter(
        strategy="time_based",
        validation_size=0.1,
        test_size=0.2,
        time_column="unix_time",
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert train_df.index.tolist() == list(range(len(train_df)))
    assert validation_df.index.tolist() == list(range(len(validation_df)))
    assert test_df.index.tolist() == list(range(len(test_df)))


def test_time_based_split_with_zero_validation_size(sample_df):
    splitter = DataSplitter(
        strategy="time_based",
        validation_size=0,
        test_size=0.2,
        time_column="unix_time",
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert len(train_df) == 16
    assert validation_df.empty
    assert len(test_df) == 4


def test_time_based_split_rejects_missing_time_column(sample_df):
    splitter = DataSplitter(
        strategy="time_based",
        validation_size=0.1,
        test_size=0.2,
        time_column="missing_time",
    )

    with pytest.raises(ValueError, match="Time column not found"):
        splitter.split(sample_df)


def test_time_based_split_ignores_stratify_column(sample_df):
    splitter = DataSplitter(
        strategy="time_based",
        validation_size=0.1,
        test_size=0.2,
        stratify_column="missing_column",
        time_column="unix_time",
    )

    train_df, validation_df, test_df = splitter.split(sample_df)

    assert len(train_df) == 14
    assert len(validation_df) == 2
    assert len(test_df) == 4