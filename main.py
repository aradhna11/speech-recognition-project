import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import MarianMTModel, MarianTokenizer
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from jiwer import wer

def load_models():
    whisper_model_name = "facebook/wav2vec2-large-960h"
    processor = Wav2Vec2Processor.from_pretrained(whisper_model_name)
    whisper_model = Wav2Vec2ForCTC.from_pretrained(whisper_model_name)
    
    translation_model_name = "Helsinki-NLP/opus-mt-en-de"
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = MarianMTModel.from_pretrained(translation_model_name)

    rag_model_name = "facebook/rag-sequence-nq"
    rag_tokenizer = RagTokenizer.from_pretrained(rag_model_name)
    retriever = RagRetriever.from_pretrained(rag_model_name, index_name="exact", passages_path="passages.txt")
    rag_model = RagSequenceForGeneration.from_pretrained(rag_model_name, retriever=retriever)
    
    return processor, whisper_model, translation_tokenizer, translation_model, rag_tokenizer, rag_model

def load_audio(file_path):
    if file_path.endswith(".wav"):
        speech_array,  sampling_rate= torchaudio.load(file_path)
    elif file_path.endswith(".mp3"):
        speech_array, sampling_rate = torchaudio.load(file_path, format="mp3")
    else:
        raise ValueError("Unsupported audio format")
    return speech_array[0].numpy()

def transcribe(audio, processor, whisper_model):
    inputs = processor(audio, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        logits = whisper_model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

def translate_text(text, translation_tokenizer, translation_model):
    translated = translation_model.generate(**translation_tokenizer(text, return_tensors="pt", padding=True))
    translated_text = [translation_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return translated_text[0]

def query_rag(translated_text, rag_tokenizer, rag_model):
    input_ids = rag_tokenizer(translated_text, return_tensors="pt").input_ids
    generated_response = rag_model.generate(input_ids=input_ids)
    response_text = rag_tokenizer.batch_decode(generated_response, skip_special_tokens=True)
    return response_text[0]

def evaluate_transcription(ground_truth, transcription):
    return wer(ground_truth, transcription)

def main():
    processor, whisper_model, translation_tokenizer, translation_model, rag_tokenizer, rag_model = load_models()
    
    audio_path = "audio.wav"  
    audio = load_audio(audio_path)
    transcription = transcribe(audio, processor, whisper_model)
    print("Transcription:", transcription)
    
    ground_truth = "Hello, my name is Aradhna Rajoria. I am currently pursuing Bachelor's of Technology in Computer Science Engineering from Amity University Gwalior."
    error = evaluate_transcription(ground_truth, transcription)
    print("Word Error Rate (WER):", error)
    
    translated_text = translate_text(transcription, translation_tokenizer, translation_model)
    print("Translated Text:", translated_text)
    
    response_text = query_rag(translated_text, rag_tokenizer, rag_model)
    print("Response from RAG:", response_text)

if __name__ == "__main__":
    main()
