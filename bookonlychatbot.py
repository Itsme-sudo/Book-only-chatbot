import os
import json
import pyttsx3
import numpy as np
from vosk import Model, KaldiRecognizer
import pyaudio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize text-to-speech
engine = pyttsx3.init()
def speak(text):
    print("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()

# Load book and split into paragraphs
def load_book(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    return paragraphs

# Compute embeddings
def embed_paragraphs(paragraphs, model):
    return model.encode(paragraphs)

# Find the best answer
def find_answer(question, paragraphs, paragraph_embeddings, model):
    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, paragraph_embeddings)
    best_idx = np.argmax(similarities)
    if similarities[0][best_idx] < 0.3:
        return "Sorry, I couldn't find the answer in the book."
    return paragraphs[best_idx]

# Initialize offline Vosk speech recognizer
def init_recognizer():
    if not os.path.exists("model"):
        raise FileNotFoundError("Vosk model folder 'model/' not found!")
    model = Model("model")
    recognizer = KaldiRecognizer(model, 16000)
    return recognizer

# Listen to user
def listen(recognizer):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()
    print("Listening...")
    while True:
        data = stream.read(4000, exception_on_overflow = False)
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")
            if text:
                return text

# Main assistant loop
def main():
    speak("Loading book and embedding model. Please wait...")
    book_file = "book.txt"
    if not os.path.exists(book_file):
        speak("Book file not found!")
        return

    paragraphs = load_book(book_file)
    model_embed = SentenceTransformer('all-MiniLM-L6-v2')
    paragraph_embeddings = embed_paragraphs(paragraphs, model_embed)
    speak(f"Book loaded with {len(paragraphs)} paragraphs.")

    recognizer = init_recognizer()
    speak("Jarvis is ready. Say your questions. Say 'exit' to quit.")

    while True:
        question = listen(recognizer)
        print("You:", question)
        if question.lower() in ["exit", "quit"]:
            speak("Goodbye!")
            break
        answer = find_answer(question, paragraphs, paragraph_embeddings, model_embed)
        speak(answer)

if __name__ == "__main__":
    main()
