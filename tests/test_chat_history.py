from __future__ import annotations

import unittest

from app.storage.chat_history import ChatConversationNotFound, ChatHistoryStore


class ChatHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = ChatHistoryStore(":memory:")

    def test_turn_is_saved_and_can_be_resumed(self):
        conversation = self.store.append_turn(
            user_id="usr_one",
            conversation_id=None,
            title="Breast cancer",
            user_message="What are common symptoms?",
            assistant_message='{"kind":"text","response":{"answer":"General education"}}',
            kind="text",
            metadata={"language": "en"},
        )
        self.assertTrue(conversation["conversation_id"].startswith("chat_"))
        self.assertEqual(conversation["message_count"], 2)
        loaded = self.store.get_conversation(
            user_id="usr_one",
            conversation_id=conversation["conversation_id"],
        )
        self.assertEqual([item["role"] for item in loaded["messages"]], ["user", "assistant"])

    def test_users_cannot_read_each_others_conversations(self):
        conversation = self.store.append_turn(
            user_id="usr_one",
            conversation_id=None,
            title="Private chat",
            user_message="Keep this private",
            assistant_message="Saved privately",
            kind="text",
        )
        with self.assertRaises(ChatConversationNotFound):
            self.store.get_conversation(
                user_id="usr_two",
                conversation_id=conversation["conversation_id"],
            )

    def test_delete_hides_history_without_deleting_other_user_data(self):
        conversation = self.store.append_turn(
            user_id="usr_one",
            conversation_id=None,
            title="To delete",
            user_message="Remove me",
            assistant_message="Saved result",
            kind="analysis",
            metadata={"model_id": "busi_breast_classifier"},
        )
        self.store.delete_conversation(
            user_id="usr_one",
            conversation_id=conversation["conversation_id"],
        )
        self.assertEqual(self.store.list_conversations(user_id="usr_one"), [])


if __name__ == "__main__":
    unittest.main()
