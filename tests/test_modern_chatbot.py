from modern.chatbot import answer

def test_known_question_returns_helpful_answer():
    assert "Transformer" in answer("What is a transformer?")

def test_blank_question_prompts_user():
    assert "Ask a question" in answer("   ")

def test_unknown_question_has_fallback():
    assert "don't know" in answer("zebra pineapple intergalactic widget")
