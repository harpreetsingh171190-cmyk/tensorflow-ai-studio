"""Small offline educational FAQ chatbot. Replace the FAQ data or retrieval method to experiment."""
from difflib import SequenceMatcher

FAQ = [
    ("What is TensorFlow?", "TensorFlow is an open-source platform for building and training machine-learning models."),
    ("What is a transformer?", "A Transformer uses attention mechanisms to model relationships between tokens in a sequence."),
    ("What is NLP?", "Natural Language Processing (NLP) helps computers process, analyze, and generate human language."),
    ("How do I train a model?", "Prepare and split your data, define a model, compile it, train with fit(), then evaluate on held-out data."),
    ("What is overfitting?", "Overfitting happens when a model fits training examples too closely and generalizes poorly. Try regularization, augmentation, or early stopping."),
    ("What is a learning rate?", "The learning rate controls how big each gradient-descent update is. Too high and training diverges; too low and it is slow or gets stuck."),
    ("What is an epoch?", "An epoch is one full pass of the training data through the model. Use validation loss and early stopping to choose how many to run."),
    ("What is data leakage?", "Data leakage is when information from the test or future data reaches training, which inflates metrics. Split first and keep duplicates or related samples in one split."),
    ("What is dropout?", "Dropout randomly switches off a fraction of units during training so the network cannot rely on any single unit, which reduces overfitting."),
    ("What is a convolutional neural network?", "A CNN slides small learned filters over an image to detect edges, textures and shapes, then combines them into higher-level features."),
    ("What is transfer learning?", "Transfer learning reuses a model pretrained on a large dataset and retrains only a small head or the top layers on your own data. It needs far less data and time."),
    ("How do I deploy a model?", "Save a model artifact, wrap inference in an API, validate inputs, add tests, then deploy behind appropriate security and monitoring."),
]

def answer(question: str) -> str:
    q = " ".join(question.lower().strip().split())
    if not q:
        return "Ask a question about TensorFlow, transformers, NLP, training, overfitting, or deployment."
    ranked = sorted(((SequenceMatcher(None, q, prompt.lower()).ratio(), response) for prompt, response in FAQ), reverse=True)
    score, response = ranked[0]
    if score < 0.30:
        return "I don't know that yet. Try asking about TensorFlow, transformers, NLP, model training, overfitting, or deployment."
    return response

if __name__ == "__main__":
    print("Harpreet Singh's ML Learning Bot (offline demo). Type 'exit' to stop.")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print("Bot:", answer(q))
