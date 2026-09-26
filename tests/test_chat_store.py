from cooking_assistant_chatbot.data import chat_store
from cooking_assistant_chatbot.data.db import get_connection


def _conn(tmp_path):
    return get_connection(str(tmp_path / "test.db"))


def test_conversations_keep_their_messages_separate(tmp_path):
    conn = _conn(tmp_path)
    a = chat_store.create_conversation(conn, "두부 요리")
    b = chat_store.create_conversation(conn, "김치찌개")

    chat_store.append_messages(conn, a, [{"role": "user", "content": "두부 요리 뭐 있어?"}])
    chat_store.append_messages(conn, b, [{"role": "user", "content": "김치찌개 칼로리"}])
    chat_store.append_messages(conn, a, [{"role": "assistant", "content": "두부조림 어때요?"}])

    assert chat_store.get_messages(conn, a) == [
        {"role": "user", "content": "두부 요리 뭐 있어?"},
        {"role": "assistant", "content": "두부조림 어때요?"},
    ]
    assert chat_store.get_messages(conn, b) == [{"role": "user", "content": "김치찌개 칼로리"}]


def test_list_is_most_recently_used_first(tmp_path):
    conn = _conn(tmp_path)
    older = chat_store.create_conversation(conn, "먼저")
    newer = chat_store.create_conversation(conn, "나중")
    chat_store.append_messages(conn, older, [{"role": "user", "content": "다시 씀"}])

    assert [c.id for c in chat_store.list_conversations(conn)] == [older, newer]


def test_delete_removes_conversation_and_its_messages(tmp_path):
    conn = _conn(tmp_path)
    keep = chat_store.create_conversation(conn, "남길 것")
    gone = chat_store.create_conversation(conn, "지울 것")
    chat_store.append_messages(conn, gone, [{"role": "user", "content": "x"}])
    chat_store.append_messages(conn, keep, [{"role": "user", "content": "y"}])

    chat_store.delete_conversation(conn, gone)

    assert [c.id for c in chat_store.list_conversations(conn)] == [keep]
    assert chat_store.get_messages(conn, gone) == []
    assert chat_store.get_messages(conn, keep) == [{"role": "user", "content": "y"}]


def test_conversations_survive_reopening_the_database(tmp_path):
    conn = _conn(tmp_path)
    conv = chat_store.create_conversation(conn, "저장 확인")
    chat_store.append_messages(conn, conv, [{"role": "user", "content": "안녕"}])
    conn.close()

    reopened = _conn(tmp_path)

    assert chat_store.list_conversations(reopened)[0].title == "저장 확인"
    assert chat_store.get_messages(reopened, conv) == [{"role": "user", "content": "안녕"}]


def test_title_from_message_trims_and_collapses_whitespace():
    assert chat_store.title_from_message("  된장찌개\n어떻게   만들어? ") == "된장찌개 어떻게 만들어?"
    long_title = chat_store.title_from_message("가" * 50)
    assert len(long_title) == 30 and long_title.endswith("…")
    assert chat_store.title_from_message("   ") == "새 채팅"
