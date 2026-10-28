"""Unit tests for the ask_user question tool."""

from deep_agent.src.agent.ask_user import ask_user, attach_ask_user

_EMPTY_QUESTION = (
    "No question was asked. Call ask_user again with a non-empty question."
)
_NO_ANSWER = "The user did not provide an answer."


class TestAskUser:
    def test_real_answer_pauses_with_cleaned_options(self, monkeypatch):
        seen: dict = {}

        def fake_interrupt(payload):
            seen["payload"] = payload
            return "  kg  "

        monkeypatch.setattr("deep_agent.src.agent.ask_user.interrupt", fake_interrupt)

        result = ask_user.invoke(
            {"question": " Units? ", "options": ["kg", "  ", "lbs"]}
        )

        assert result == "kg"
        assert seen["payload"] == {
            "type": "user_question",
            "question": "Units?",
            "options": ["kg", "lbs"],
        }

    def test_blank_question_does_not_pause(self, monkeypatch):
        def fake_interrupt(payload):
            raise AssertionError(f"blank question must not pause: {payload}")

        monkeypatch.setattr("deep_agent.src.agent.ask_user.interrupt", fake_interrupt)

        assert ask_user.invoke({"question": "   "}) == _EMPTY_QUESTION

    def test_blank_options_are_dropped(self, monkeypatch):
        seen: dict = {}

        def fake_interrupt(payload):
            seen["payload"] = payload
            return "stone"

        monkeypatch.setattr("deep_agent.src.agent.ask_user.interrupt", fake_interrupt)

        result = ask_user.invoke({"question": "Units?", "options": ["  ", ""]})

        assert result == "stone"
        assert seen["payload"]["options"] == []

    def test_non_text_answer_is_reported(self, monkeypatch):
        monkeypatch.setattr(
            "deep_agent.src.agent.ask_user.interrupt",
            lambda payload: {"answer": "kg"},
        )

        assert ask_user.invoke({"question": "Units?"}) == _NO_ANSWER

    def test_description_says_when_to_ask_and_when_to_omit_options(self):
        description = ask_user.description
        assert "ask_user" in description
        assert "omit options" in description
        assert "complete answer" in description

    def test_attach_adds_the_tool_once(self):
        once = attach_ask_user([])
        twice = attach_ask_user(once)

        assert [tool.name for tool in once] == ["ask_user"]
        assert [tool.name for tool in twice] == ["ask_user"]
        assert twice[0] is ask_user
