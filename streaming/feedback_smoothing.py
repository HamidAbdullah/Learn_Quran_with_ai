"""
Phase 4.1 — Feedback smoothing for streaming word_feedback.

Keeps the last 2 partial word_feedback states. A word is only marked "wrong"
(or "minor_mistake") if it was already wrong/minor in the previous window,
reducing flicker between correct and wrong as the partial transcript updates.
"""

from collections import deque
from typing import Any, Dict, List


# Statuses we require to appear in 2 consecutive windows before showing
CONFIRM_BEFORE_SHOW = ("wrong", "minor_mistake")


class WordFeedbackSmoother:
    """
    Maintains last 2 word_feedback lists; smooths so wrong/minor_mistake
    only show after 2 consecutive windows with that status.
    """

    def __init__(self, history_size: int = 2):
        """
        Args:
            history_size: Number of previous states to keep (default 2).
        """
        self._history: deque = deque(maxlen=max(1, history_size))

    def update(self, word_feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Take the latest word_feedback from alignment; return a smoothed copy.

        Rule: for each word, if the new status is "wrong" or "minor_mistake",
        only emit that status if the same word had a "wrong" or "minor_mistake"
        status in the previous window. Otherwise emit the previous status
        (or "correct" if no previous), so we don't flicker to wrong on one bad window.

        Args:
            word_feedback: List of {"word": str, "status": str} from get_lightweight_word_feedback.

        Returns:
            New list with same structure; status may be smoothed.
        """
        if not word_feedback:
            self._history.append([])
            return []

        prev_list = list(self._history)[-1] if self._history else []
        self._history.append(word_feedback)

        smoothed: List[Dict[str, Any]] = []
        for i, item in enumerate(word_feedback):
            word = item.get("word", "")
            status = item.get("status", "pending")
            prev_status = None
            if i < len(prev_list):
                prev_status = prev_list[i].get("status", "pending")

            if status in CONFIRM_BEFORE_SHOW:
                # Only show wrong/minor if it was already wrong/minor last time
                if prev_status in CONFIRM_BEFORE_SHOW:
                    new_status = status
                else:
                    new_status = prev_status if prev_status else "correct"
            else:
                new_status = status

            smoothed.append({"word": word, "status": new_status})
        return smoothed

    def reset(self) -> None:
        """Clear history (e.g. on new verse or reconnect)."""
        self._history.clear()
