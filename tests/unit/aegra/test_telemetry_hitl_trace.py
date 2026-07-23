"""Unit tests for Langfuse HITL single-flow (open-root) tracing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from deep_agent.aegra import telemetry as telemetry_mod
from deep_agent.aegra.telemetry import (
    LangfuseObservabilityProvider,
    _build_hitl_aware_handler_class,
)


@pytest.fixture()
def hitl_handler_class():
    """Build the HITL-aware handler class (requires langfuse installed)."""
    return _build_hitl_aware_handler_class()


def _mock_root_obs() -> MagicMock:
    obs = MagicMock()
    obs.trace_id = "a" * 32
    obs.id = "b" * 16
    obs._otel_span = MagicMock()
    return obs


class TestHitlAwareCallbackHandler:
    def test_interrupt_keeps_root_span_open(self, hitl_handler_class):
        """Nested GraphInterrupt must not end the root orchestrator observation."""
        pytest.importorskip("langgraph")
        from langgraph.errors import GraphInterrupt

        handler = hitl_handler_class()
        root_id, child_id = uuid4(), uuid4()
        handler._track_run(
            run_id=root_id, parent_run_id=None, metadata={"thread_id": "t1"}
        )
        handler._track_run(
            run_id=child_id, parent_run_id=root_id, metadata={"thread_id": "t1"}
        )
        root_obs = _mock_root_obs()
        handler._runs[root_id] = root_obs

        handler.on_chain_error(
            GraphInterrupt(()), run_id=child_id, parent_run_id=root_id
        )
        assert handler._open_hitl_roots["t1"] is root_obs
        assert root_id in handler._keep_open_root_run_ids

        handler.on_chain_end({"ok": True}, run_id=root_id, parent_run_id=None)

        root_obs.end.assert_not_called()
        assert handler._open_hitl_roots["t1"] is root_obs
        assert root_id not in handler._runs  # detached from this invoke
        assert root_id not in handler._keep_open_root_run_ids

    def test_resume_rebinds_to_open_root_without_new_span(self, hitl_handler_class):
        """Command(resume=...) must reuse the open observation, not start a new root."""
        pytest.importorskip("langgraph")
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        handler = hitl_handler_class()
        root_id, child_id, resume_id = uuid4(), uuid4(), uuid4()
        handler._track_run(
            run_id=root_id, parent_run_id=None, metadata={"thread_id": "t1"}
        )
        handler._track_run(
            run_id=child_id, parent_run_id=root_id, metadata={"thread_id": "t1"}
        )
        root_obs = _mock_root_obs()
        handler._runs[root_id] = root_obs

        handler.on_chain_error(
            GraphInterrupt(()), run_id=child_id, parent_run_id=root_id
        )
        handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

        with (
            patch.object(handler, "_attach_observation") as attach,
            patch.object(
                handler,
                "_langfuse_client",
                MagicMock(start_observation=MagicMock()),
            ),
        ):
            # Make attach actually bind like production for the assertion below.
            def _attach(run_id, observation):
                handler._runs[run_id] = observation

            attach.side_effect = _attach

            handler.on_chain_start(
                {"name": "orchestrator"},
                Command(resume={"decisions": [{"type": "approve"}]}),
                run_id=resume_id,
                parent_run_id=None,
                metadata={"thread_id": "t1"},
            )

            attach.assert_called_once()
            assert attach.call_args.args[1] is root_obs
            handler._langfuse_client.start_observation.assert_not_called()
            assert handler._runs[resume_id] is root_obs
            # Ownership moves to _runs; map must not keep a zombie handle.
            assert "t1" not in handler._open_hitl_roots

    def test_final_end_closes_open_root(self, hitl_handler_class):
        """A completing root invoke must end the open observation and clear state."""
        pytest.importorskip("langgraph")
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        handler = hitl_handler_class()
        root_id, child_id, resume_id = uuid4(), uuid4(), uuid4()
        handler._track_run(
            run_id=root_id, parent_run_id=None, metadata={"thread_id": "t1"}
        )
        handler._track_run(
            run_id=child_id, parent_run_id=root_id, metadata={"thread_id": "t1"}
        )
        root_obs = _mock_root_obs()
        handler._runs[root_id] = root_obs

        handler.on_chain_error(
            GraphInterrupt(()), run_id=child_id, parent_run_id=root_id
        )
        handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

        def _attach(run_id, observation):
            handler._runs[run_id] = observation
            handler._context_tokens[run_id] = MagicMock()

        with patch.object(handler, "_attach_observation", side_effect=_attach):
            handler.on_chain_start(
                {"name": "orchestrator"},
                Command(resume={"decisions": [{"type": "approve"}]}),
                run_id=resume_id,
                parent_run_id=None,
                metadata={"thread_id": "t1"},
            )

        handler.on_chain_end({"done": True}, run_id=resume_id, parent_run_id=None)

        root_obs.end.assert_called()
        assert "t1" not in handler._open_hitl_roots

    def test_fresh_message_ends_stale_open_root(self, hitl_handler_class):
        """A new user message must close a leftover open HITL root for that thread."""
        handler = hitl_handler_class()
        stale = _mock_root_obs()
        handler._open_hitl_roots["t1"] = stale

        with patch.object(
            handler.__class__.__bases__[0],
            "on_chain_start",
            return_value=None,
        ) as super_start:
            handler.on_chain_start(
                {"name": "orchestrator"},
                {"messages": [{"role": "user", "content": "hi"}]},
                run_id=uuid4(),
                parent_run_id=None,
                metadata={"thread_id": "t1"},
            )
            super_start.assert_called_once()

        stale.end.assert_called_once()
        assert "t1" not in handler._open_hitl_roots

    def test_separate_handler_instances_do_not_share_open_roots(
        self, hitl_handler_class
    ):
        """Documents why setup_langfuse_tracing must reuse one handler instance."""
        a = hitl_handler_class()
        b = hitl_handler_class()
        obs = _mock_root_obs()
        a._open_hitl_roots["t"] = obs
        assert "t" not in b._open_hitl_roots

    def test_parent_command_does_not_keep_root_open(self, hitl_handler_class):
        """Nested ParentCommand must not mark the orchestrator as an open HITL root."""
        pytest.importorskip("langgraph")
        from langgraph.errors import ParentCommand
        from langgraph.types import Command

        handler = hitl_handler_class()
        root_id, child_id = uuid4(), uuid4()
        handler._track_run(
            run_id=root_id, parent_run_id=None, metadata={"thread_id": "t1"}
        )
        handler._track_run(
            run_id=child_id, parent_run_id=root_id, metadata={"thread_id": "t1"}
        )
        root_obs = _mock_root_obs()
        handler._runs[root_id] = root_obs

        handler.on_chain_error(
            ParentCommand(Command()), run_id=child_id, parent_run_id=root_id
        )

        assert "t1" not in handler._open_hitl_roots
        assert root_id not in handler._keep_open_root_run_ids

    def test_resume_rebind_failure_falls_back_to_super(self, hitl_handler_class):
        """If open-root rebind fails, resume must still open a normal Langfuse root."""
        pytest.importorskip("langgraph")
        from langgraph.types import Command

        handler = hitl_handler_class()
        resume_id = uuid4()
        open_obs = _mock_root_obs()
        handler._open_hitl_roots["t1"] = open_obs

        with (
            patch.object(
                handler, "_attach_observation", side_effect=RuntimeError("boom")
            ),
            patch.object(
                handler.__class__.__bases__[0],
                "on_chain_start",
                return_value=None,
            ) as super_start,
        ):
            handler.on_chain_start(
                {"name": "orchestrator"},
                Command(resume={"decisions": [{"type": "approve"}]}),
                run_id=resume_id,
                parent_run_id=None,
                metadata={"thread_id": "t1"},
            )

            super_start.assert_called_once()

        open_obs.end.assert_called_once()
        assert "t1" not in handler._open_hitl_roots

    def test_open_hitl_roots_evicts_oldest_when_over_cap(self, hitl_handler_class):
        """Abandoned open roots must be ended once the LRU cap is exceeded."""
        handler = hitl_handler_class()
        oldest = _mock_root_obs()
        middle = _mock_root_obs()
        newest = _mock_root_obs()

        with patch.object(telemetry_mod, "_MAX_OPEN_HITL_ROOTS", 2):
            handler._store_open_hitl_root("t1", oldest)
            handler._store_open_hitl_root("t2", middle)
            handler._store_open_hitl_root("t3", newest)

        assert "t1" not in handler._open_hitl_roots
        assert handler._open_hitl_roots["t2"] is middle
        assert handler._open_hitl_roots["t3"] is newest
        oldest.end.assert_called_once()
        middle.end.assert_not_called()
        newest.end.assert_not_called()

    def test_store_replaces_different_obs_ends_previous(self, hitl_handler_class):
        """Overwriting the same thread key must end a different prior observation."""
        handler = hitl_handler_class()
        first = _mock_root_obs()
        second = _mock_root_obs()
        handler._store_open_hitl_root("t1", first)
        handler._store_open_hitl_root("t1", second)

        first.end.assert_called_once()
        second.end.assert_not_called()
        assert handler._open_hitl_roots["t1"] is second

    def test_resume_hard_error_clears_open_root_for_next_resume(
        self, hitl_handler_class
    ):
        """Hard error after approve must not leave a dead obs for the next resume."""
        pytest.importorskip("langgraph")
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        handler = hitl_handler_class()
        root_id, child_id, resume_id, resume2_id = (
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
        )
        handler._track_run(
            run_id=root_id, parent_run_id=None, metadata={"thread_id": "t1"}
        )
        handler._track_run(
            run_id=child_id, parent_run_id=root_id, metadata={"thread_id": "t1"}
        )
        root_obs = _mock_root_obs()
        handler._runs[root_id] = root_obs

        handler.on_chain_error(
            GraphInterrupt(()), run_id=child_id, parent_run_id=root_id
        )
        handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

        def _attach(run_id, observation):
            handler._runs[run_id] = observation
            handler._context_tokens[run_id] = MagicMock()

        with patch.object(handler, "_attach_observation", side_effect=_attach):
            handler.on_chain_start(
                {"name": "orchestrator"},
                Command(resume={"decisions": [{"type": "approve"}]}),
                run_id=resume_id,
                parent_run_id=None,
                metadata={"thread_id": "t1"},
            )

        assert "t1" not in handler._open_hitl_roots

        # Simulate a leftover map entry (pre-pop bug / race), then root hard error.
        handler._open_hitl_roots["t1"] = root_obs
        handler.on_chain_error(
            RuntimeError("tool failed"), run_id=resume_id, parent_run_id=None
        )
        assert "t1" not in handler._open_hitl_roots

        with (
            patch.object(handler, "_attach_observation") as attach,
            patch.object(
                handler.__class__.__bases__[0],
                "on_chain_start",
                return_value=None,
            ) as super_start,
        ):
            handler.on_chain_start(
                {"name": "orchestrator"},
                Command(resume={"decisions": [{"type": "approve"}]}),
                run_id=resume2_id,
                parent_run_id=None,
                metadata={"thread_id": "t1"},
            )
            attach.assert_not_called()
            super_start.assert_called_once()

    def test_second_hitl_in_same_thread_reopens_after_rebind(self, hitl_handler_class):
        """Pop-on-rebind must still allow a later nested GraphInterrupt to keep-open."""
        pytest.importorskip("langgraph")
        from langgraph.errors import GraphInterrupt
        from langgraph.types import Command

        handler = hitl_handler_class()
        root_id, child_id, resume_id, child2_id = (
            uuid4(),
            uuid4(),
            uuid4(),
            uuid4(),
        )
        handler._track_run(
            run_id=root_id, parent_run_id=None, metadata={"thread_id": "t1"}
        )
        handler._track_run(
            run_id=child_id, parent_run_id=root_id, metadata={"thread_id": "t1"}
        )
        root_obs = _mock_root_obs()
        handler._runs[root_id] = root_obs

        handler.on_chain_error(
            GraphInterrupt(()), run_id=child_id, parent_run_id=root_id
        )
        handler.on_chain_end({}, run_id=root_id, parent_run_id=None)

        def _attach(run_id, observation):
            handler._runs[run_id] = observation
            handler._context_tokens[run_id] = MagicMock()

        with patch.object(handler, "_attach_observation", side_effect=_attach):
            handler.on_chain_start(
                {"name": "orchestrator"},
                Command(resume={"decisions": [{"type": "approve"}]}),
                run_id=resume_id,
                parent_run_id=None,
                metadata={"thread_id": "t1"},
            )

        assert "t1" not in handler._open_hitl_roots
        assert handler._runs[resume_id] is root_obs

        handler._track_run(
            run_id=child2_id, parent_run_id=resume_id, metadata={"thread_id": "t1"}
        )
        handler.on_chain_error(
            GraphInterrupt(()), run_id=child2_id, parent_run_id=resume_id
        )
        assert handler._open_hitl_roots["t1"] is root_obs
        assert resume_id in handler._keep_open_root_run_ids

        handler.on_chain_end({}, run_id=resume_id, parent_run_id=None)
        root_obs.end.assert_not_called()
        assert handler._open_hitl_roots["t1"] is root_obs


class TestLangfuseObservabilityProviderMetadata:
    def test_includes_thread_id_for_resume_key(self):
        provider = LangfuseObservabilityProvider()
        with patch.object(
            LangfuseObservabilityProvider,
            "is_enabled",
            return_value=True,
        ):
            metadata = provider.get_metadata(
                run_id="run-1",
                thread_id="thread-abc",
                user_identity=None,
            )

        assert metadata["langfuse_session_id"] == "thread-abc"
        assert metadata["thread_id"] == "thread-abc"
        assert "langfuse_trace_name" in metadata
