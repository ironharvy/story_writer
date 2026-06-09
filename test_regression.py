"""The regression eval the pipeline must pass: every gold negative is caught by
its labelled metric, and every good case stays clean."""
import canon as canon_mod
from bench import regression

CANON = canon_mod.load_canon(canon_mod.DEFAULT_CANON_PATH)


def test_every_gold_negative_is_caught():
    for case in regression.GOLD_NEGATIVES:
        passed, detail = regression.evaluate_case(case, CANON)
        assert passed, f"{case.id}: {detail}"


def test_every_good_case_stays_clean():
    for case in regression.GOOD_CASES:
        passed, detail = regression.evaluate_case(case, CANON)
        assert passed, f"{case.id}: {detail}"


def test_run_reports_all_pass():
    all_passed, rows = regression.run(CANON)
    assert all_passed, [r for r in rows if not r[1]]


def test_alias_good_case_would_fail_without_normalization():
    # Sanity: the alias case is only clean BECAUSE accepted_aliases normalizes it;
    # without normalization the spelled-out form drifts from canonical "Vessel 84".
    import qa
    case = next(c for c in regression.GOOD_CASES if c.id == "alias_eighty_four")
    raw = qa.check_name_drift(case.text)
    assert any(f.check == "name_drift" and f.severity == "warn" for f in raw)
