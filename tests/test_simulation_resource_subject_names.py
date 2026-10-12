from walinet.training_data.simulation_resources import _subject_names_match


def test_current_and_legacy_subject_names_match() -> None:
    assert _subject_names_match(
        requested_subject="Brisbane/Vol3",
        stored_subject="Vol3_Brisbane",
    )
    assert _subject_names_match(
        requested_subject="UCSF/Vol3",
        stored_subject="Vol3_UCSF",
    )
    assert _subject_names_match(
        requested_subject="Vienna/Vol5",
        stored_subject="Vol5",
    )
    assert _subject_names_match(
        requested_subject="MS/MS_180",
        stored_subject="MS_180",
    )


def test_unrelated_subject_names_do_not_match() -> None:
    assert not _subject_names_match(
        requested_subject="Brisbane/Vol3",
        stored_subject="Vol4_Brisbane",
    )
    assert not _subject_names_match(
        requested_subject="Brisbane/Vol3",
        stored_subject="Vol3_London",
    )
    assert not _subject_names_match(
        requested_subject="Brisbane/Vol3",
        stored_subject="unrelated",
    )
