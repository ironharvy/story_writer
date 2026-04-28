"""Tests for qa.py quality-assurance checks."""

from types import SimpleNamespace
from unittest.mock import patch

from qa import run_quality_checks


def _make_args(check_similar: bool = True, threshold: float = 0.65) -> SimpleNamespace:
    """Build a minimal args namespace for QA tests."""
    return SimpleNamespace(check_similar=check_similar, similar_threshold=threshold)


SAMPLE_TEXT = (
    "The knight rode through the dark forest. Birds scattered from the treetops. "
    "The knight rode through the shadowy forest. A distant horn echoed."
)


class TestRunQualityChecks:
    """Tests for the top-level run_quality_checks dispatcher."""

    @patch("qa._run_similarity_check")
    def test_delegates_to_similarity_check(self, mock_sim):
        args = _make_args(check_similar=True, threshold=0.7)
        run_quality_checks(args=args, final_story_text="some text")
        mock_sim.assert_called_once_with(
            check_enabled=True,
            threshold=0.7,
            final_story_text="some text",
        )

    @patch("qa._run_similarity_check")
    def test_passes_disabled_flag(self, mock_sim):
        args = _make_args(check_similar=False)
        run_quality_checks(args=args, final_story_text="text")
        mock_sim.assert_called_once()
        assert mock_sim.call_args.kwargs["check_enabled"] is False


class TestRunSimilarityCheck:
    """Tests for _run_similarity_check behaviour."""

    @patch("qa.ui")
    def test_skips_when_disabled(self, mock_ui):
        args = _make_args(check_similar=False)
        run_quality_checks(args=args, final_story_text=SAMPLE_TEXT)
        mock_ui.print_similarity_report.assert_not_called()

    @patch("qa.ui")
    def test_runs_and_prints_report(self, mock_ui):
        args = _make_args(check_similar=True, threshold=0.5)
        run_quality_checks(args=args, final_story_text=SAMPLE_TEXT)
        mock_ui.print_similarity_report.assert_called_once()
        report_text = mock_ui.print_similarity_report.call_args[0][0]
        assert isinstance(report_text, str)

    @patch("qa.ui")
    def test_logs_warning_when_similar_pairs_found(self, mock_ui):
        args = _make_args(check_similar=True, threshold=0.5)
        with patch("qa.logger") as mock_logger:
            run_quality_checks(args=args, final_story_text=SAMPLE_TEXT)
            mock_logger.warning.assert_called_once()
            assert "similar sentence pair" in mock_logger.warning.call_args[0][0].lower()

    @patch("qa.ui")
    def test_no_warning_when_no_similar_pairs(self, mock_ui):
        distinct_text = "The sky is blue. Cats are fluffy. I like pizza."
        args = _make_args(check_similar=True, threshold=0.99)
        with patch("qa.logger") as mock_logger:
            run_quality_checks(args=args, final_story_text=distinct_text)
            mock_logger.warning.assert_not_called()
