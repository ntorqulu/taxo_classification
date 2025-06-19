import pytest
import random
from constants.taxonomy_labels import TAXONOMY_LABELS, wrong_class_values


def test_wrong_class_values_all_ok():
    for level_name in TAXONOMY_LABELS.keys():
        result = wrong_class_values(level_name, TAXONOMY_LABELS[level_name])
        assert result is None


def test_wrong_class_values_non_existent_level():
    with pytest.raises(ValueError):
        wrong_class_values("non_existent_level", TAXONOMY_LABELS[list(TAXONOMY_LABELS.keys())[0]])


def test_wrong_class_values_missing():
    for level_name in TAXONOMY_LABELS.keys():
        values = TAXONOMY_LABELS[level_name].copy()
        idx = random.randint(0, len(values)-1)
        missing_value = values.pop(idx)
        result = wrong_class_values(level_name, values)
        assert result is not None
        assert result['unknown'] == []
        assert result['missing'] == [missing_value]


def test_wrong_class_values_unknown():
    for level_name in TAXONOMY_LABELS.keys():
        values = TAXONOMY_LABELS[level_name].copy()
        idx = random.randint(0, len(values) - 1)
        values.insert(idx, 'non_existent_value')
        result = wrong_class_values(level_name, values)
        assert result is not None
        assert result['unknown'] == ['non_existent_value']
        assert result['missing'] == []
