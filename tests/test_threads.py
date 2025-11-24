"""Tests for the threads route."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from template_agent.src.routes.threads import list_threads, router


class TestThreadsEndpoint:
    """Test cases for the threads endpoint."""

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.psycopg2")
    async def test_list_threads_postgresql_success(self, mock_psycopg2, mock_settings):
        """Test successful thread listing from PostgreSQL."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        # Mock database connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn

        # Mock database response
        mock_cursor.fetchall.return_value = [
            ("thread_123",),
            ("thread_456",),
            ("thread_789",),
        ]

        # Call the endpoint
        result = await list_threads("user_123")

        # Verify result
        assert result == ["thread_123", "thread_456", "thread_789"]

        # Verify database query
        mock_cursor.execute.assert_called_once()
        query = mock_cursor.execute.call_args[0][0]
        assert "SELECT distinct thread_id FROM checkpoints" in query
        assert "metadata->>'user_id'='user_123'" in query

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.psycopg2")
    async def test_list_threads_postgresql_no_threads(
        self, mock_psycopg2, mock_settings
    ):
        """Test thread listing with no threads found."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn

        # No threads found
        mock_cursor.fetchall.return_value = []

        result = await list_threads("user_with_no_threads")

        assert result == []

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.psycopg2")
    async def test_list_threads_postgresql_connection_error(
        self, mock_psycopg2, mock_settings
    ):
        """Test PostgreSQL connection error handling."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_psycopg2.connect.side_effect = Exception("Connection failed")

        with pytest.raises(HTTPException) as exc_info:
            await list_threads("user_123")

        assert exc_info.value.status_code == 500
        assert "Unexpected error" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.get_user_threads")
    async def test_list_threads_inmemory_success(
        self, mock_get_user_threads, mock_settings
    ):
        """Test successful thread listing from in-memory storage."""
        mock_settings.USE_INMEMORY_SAVER = True

        # Mock thread registry response
        mock_get_user_threads.return_value = ["thread_abc", "thread_def"]

        result = await list_threads("user_456")

        assert result == ["thread_abc", "thread_def"]
        mock_get_user_threads.assert_called_once_with("user_456")

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.get_user_threads")
    async def test_list_threads_inmemory_empty(
        self, mock_get_user_threads, mock_settings
    ):
        """Test in-memory thread listing with no threads."""
        mock_settings.USE_INMEMORY_SAVER = True

        mock_get_user_threads.return_value = []

        result = await list_threads("user_no_threads")

        assert result == []

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.get_user_threads")
    async def test_list_threads_inmemory_error(
        self, mock_get_user_threads, mock_settings
    ):
        """Test error handling for in-memory storage."""
        mock_settings.USE_INMEMORY_SAVER = True

        mock_get_user_threads.side_effect = Exception("Registry error")

        with pytest.raises(HTTPException) as exc_info:
            await list_threads("user_123")

        assert exc_info.value.status_code == 500
        assert "Failed to retrieve threads from registry" in exc_info.value.detail

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.app_logger")
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.get_user_threads")
    async def test_list_threads_logging_inmemory(
        self, mock_get_user_threads, mock_settings, mock_logger
    ):
        """Test that thread listing logs appropriately for in-memory storage."""
        mock_settings.USE_INMEMORY_SAVER = True

        mock_get_user_threads.return_value = ["thread1", "thread2"]

        await list_threads("user_123")

        # Verify logging
        log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any(
            "Using in-memory storage - retrieving threads from registry" in msg
            for msg in log_messages
        )
        assert any("Found 2 threads in registry" in msg for msg in log_messages)

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.app_logger")
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.psycopg2")
    async def test_list_threads_logging_postgresql(
        self, mock_psycopg2, mock_settings, mock_logger
    ):
        """Test that thread listing logs appropriately for PostgreSQL."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn

        mock_cursor.fetchall.return_value = [("thread1",), ("thread2",), ("thread3",)]

        await list_threads("user_456")

        # Verify logging
        log_messages = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any(
            "Found 3 threads for user_id: user_456" in msg for msg in log_messages
        )


class TestThreadsIntegration:
    """Integration tests for the threads route."""

    def test_threads_route_registered(self):
        """Test that threads route is registered correctly."""
        routes = [route.path for route in router.routes]
        assert "/v1/threads/{user_id}" in routes

    def test_threads_route_methods(self):
        """Test that threads route accepts GET method."""
        for route in router.routes:
            if "/v1/threads/" in route.path:
                assert "GET" in route.methods

    def test_threads_with_test_client_inmemory(self):
        """Test threads endpoint with FastAPI test client using in-memory storage."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("template_agent.src.routes.threads.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = True
            with patch(
                "template_agent.src.routes.threads.get_user_threads"
            ) as mock_get:
                mock_get.return_value = ["thread_1", "thread_2"]

                client = TestClient(app)
                response = client.get("/v1/threads/user_123")

                assert response.status_code == 200
                data = response.json()
                assert data == ["thread_1", "thread_2"]

    def test_threads_with_test_client_postgresql(self):
        """Test threads endpoint with FastAPI test client using PostgreSQL."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("template_agent.src.routes.threads.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = False
            mock_settings.database_uri = "postgresql://test"
            with patch("template_agent.src.routes.threads.psycopg2") as mock_psycopg2:
                mock_conn = MagicMock()
                mock_cursor = MagicMock()
                mock_conn.cursor.return_value = mock_cursor
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=None)
                mock_psycopg2.connect.return_value = mock_conn
                mock_cursor.fetchall.return_value = [("thread_a",), ("thread_b",)]

                client = TestClient(app)
                response = client.get("/v1/threads/user_456")

                assert response.status_code == 200
                data = response.json()
                assert data == ["thread_a", "thread_b"]

    def test_threads_error_response(self):
        """Test that threads endpoint returns proper error response."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("template_agent.src.routes.threads.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = False
            mock_settings.database_uri = "postgresql://test"
            with patch("template_agent.src.routes.threads.psycopg2") as mock_psycopg2:
                mock_psycopg2.connect.side_effect = Exception("Database error")

                client = TestClient(app)
                response = client.get("/v1/threads/user_123")

                assert response.status_code == 500
                data = response.json()
                assert "detail" in data

    def test_threads_with_special_characters_in_user_id(self):
        """Test threads endpoint with special characters in user_id."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        with patch("template_agent.src.routes.threads.settings") as mock_settings:
            mock_settings.USE_INMEMORY_SAVER = True
            with patch(
                "template_agent.src.routes.threads.get_user_threads"
            ) as mock_get:
                mock_get.return_value = []

                client = TestClient(app)
                # Test with URL-encoded special characters
                response = client.get("/v1/threads/user%40example.com")

                assert response.status_code == 200
                # Verify the decoded user_id was passed
                mock_get.assert_called_with("user@example.com")


class TestThreadsSQLInjectionPrevention:
    """Test cases for SQL injection prevention in threads route."""

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.psycopg2")
    async def test_sql_injection_prevention(self, mock_psycopg2, mock_settings):
        """Test that the query is vulnerable to SQL injection (as written)."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn
        mock_cursor.fetchall.return_value = []

        # Attempt SQL injection
        malicious_user_id = "'; DROP TABLE checkpoints; --"

        await list_threads(malicious_user_id)

        # Check the actual query executed
        # NOTE: The current implementation uses f-string formatting which is vulnerable
        # This test documents the vulnerability
        query = mock_cursor.execute.call_args[0][0]
        assert (
            malicious_user_id in query
        )  # The malicious input is directly in the query

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.psycopg2")
    async def test_threads_with_quotes_in_user_id(self, mock_psycopg2, mock_settings):
        """Test handling of quotes in user_id."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn
        mock_cursor.fetchall.return_value = []

        user_id_with_quotes = "user's_id"

        await list_threads(user_id_with_quotes)

        # The query will include the quotes directly
        query = mock_cursor.execute.call_args[0][0]
        assert user_id_with_quotes in query


class TestThreadsPerformance:
    """Test cases for threads route performance considerations."""

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.get_user_threads")
    async def test_large_number_of_threads_inmemory(
        self, mock_get_user_threads, mock_settings
    ):
        """Test handling of large number of threads in memory."""
        mock_settings.USE_INMEMORY_SAVER = True

        # Create a large list of threads
        large_thread_list = [f"thread_{i}" for i in range(1000)]
        mock_get_user_threads.return_value = large_thread_list

        result = await list_threads("user_with_many_threads")

        assert len(result) == 1000
        assert result[0] == "thread_0"
        assert result[999] == "thread_999"

    @pytest.mark.asyncio
    @patch("template_agent.src.routes.threads.settings")
    @patch("template_agent.src.routes.threads.psycopg2")
    async def test_large_number_of_threads_postgresql(
        self, mock_psycopg2, mock_settings
    ):
        """Test handling of large number of threads from PostgreSQL."""
        mock_settings.USE_INMEMORY_SAVER = False
        mock_settings.database_uri = "postgresql://test"

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=None)
        mock_psycopg2.connect.return_value = mock_conn

        # Create a large result set
        large_result = [(f"thread_{i}",) for i in range(5000)]
        mock_cursor.fetchall.return_value = large_result

        result = await list_threads("user_with_many_threads")

        assert len(result) == 5000
        assert result[0] == "thread_0"
        assert result[4999] == "thread_4999"
