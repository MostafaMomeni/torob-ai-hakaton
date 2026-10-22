import pytest
from shopping_assistant_data.repository import DataRepository


def test_load_and_references():
    repo = DataRepository()
    repo.load_all()
    # basic counts
    assert len(repo._products) > 100
    assert len(repo._shops) > 100
    assert len(repo._offers) > 100
    refs = repo.validate_references()
    # there may be missing shops or products but offers should not all be missing
    assert isinstance(refs, dict)


def test_coordinate_validation():
    repo = DataRepository()
    repo.load_all()
    coords = repo.shops_with_valid_coords()
    assert all(s.lat is not None and s.lon is not None for s in coords)


def test_duplicates():
    repo = DataRepository()
    repo.load_all()
    dups = repo.find_duplicate_shops()
    assert isinstance(dups, list)
