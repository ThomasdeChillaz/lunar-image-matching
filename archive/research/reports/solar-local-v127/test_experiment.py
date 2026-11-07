from experiment import roster
from solar_evaluation_v70 import bounds_disjoint


def test_same_case_scope_includes_all_seed_pairs_and_certified_negatives():
    inputs, labels, rows, _, _, libraries = roster()
    assert len(inputs) == len(labels) == len(libraries) == 4
    assert len(rows) == 32 and sum(r["positive"] for r in rows) == 16
    assert len({(r["reference_index"], r["query_index"], r["reference_seed"], r["query_seed"]) for r in rows}) == 32
    assert all(len(library) == 24 for library in libraries.values())
    for row in rows:
        if not row["positive"]:
            proof = row["nonoverlap"]
            assert proof["full_image_footprints"]
            assert row["component"] != row["query_component"]
            assert bounds_disjoint(proof["reference_bounds"], proof["query_bounds"])
