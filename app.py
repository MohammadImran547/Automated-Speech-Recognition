import whisper
import torch
import gradio as gr
import translators as ts
from typing import Dict

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper model
transcribe_model = whisper.load_model("base").to(device)

# Language mappings (replacing googletrans.LANGUAGES)
LANGUAGES = {
    "en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "ur": "Urdu", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ne": "Nepali", "sd": "Sindhi",
    "sa": "Sanskrit", "fa": "Persian", "ar": "Arabic", "fr": "French", "es": "Spanish",
    "de": "German", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "af": "Afrikaans", "sq": "Albanian", "am": "Amharic", "hy": "Armenian",
    "az": "Azerbaijani", "eu": "Basque", "be": "Belarusian", "bs": "Bosnian",
    "bg": "Bulgarian", "ca": "Catalan", "hr": "Croatian", "cs": "Czech",
    "da": "Danish", "nl": "Dutch", "eo": "Esperanto", "et": "Estonian",
    "tl": "Filipino", "fi": "Finnish", "ka": "Georgian", "el": "Greek",
    "ht": "Haitian Creole", "iw": "Hebrew", "hu": "Hungarian", "is": "Icelandic",
    "id": "Indonesian", "ga": "Irish", "it": "Italian", "jw": "Javanese",
    "kk": "Kazakh", "la": "Latin", "lv": "Latvian", "lt": "Lithuanian",
    "mk": "Macedonian", "ms": "Malay", "mn": "Mongolian", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sr": "Serbian", "sk": "Slovak", "sl": "Slovenian", "sw": "Swahili",
    "sv": "Swedish", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian"
}

# Indian languages subset
indian = {
    "bengali": "bn", "english": "en", "gujarati": "gu", "hindi": "hi", 
    "kannada": "kn", "malayalam": "ml", "marathi": "mr", "nepali": "ne",
    "odia": "or", "punjabi": "pa", "sindhi": "sd", "tamil": "ta",
    "telugu": "te", "urdu": "ur"
}

# Other languages
outers = {
    lang_name: lang_code 
    for lang_code, lang_name in LANGUAGES.items() 
    if lang_code not in indian.values()
}

def transcribe_and_detect_language(audio_path: str) -> tuple:
    """Transcribe audio and detect language using Whisper"""
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(transcribe_model.device)
    
    # Detect language
    _, probs = transcribe_model.detect_language(mel)
    detected_lang_code = max(probs, key=probs.get)
    detected_lang = LANGUAGES.get(detected_lang_code, detected_lang_code)
    
    # Transcribe
    result = transcribe_model.transcribe(audio_path)
    return detected_lang, result["text"]

def translate(text_language: str, text: str, translate_in_language: str) -> str:
    """Translate text using translators library"""
    try:
        # Create reverse mapping for language lookup
        lang_map = {v.lower(): k for k, v in LANGUAGES.items()}
        
        # Get language codes
        src_lang = lang_map.get(text_language.lower(), text_language.lower())
        tgt_lang = lang_map.get(translate_in_language.lower(), translate_in_language.lower())
        
        if not src_lang or not tgt_lang:
            return "Unsupported language"
            
        # Use Google translator through translators library
        translated = ts.translate_text(
            text,
            translator='google',
            from_language=src_lang,
            to_language=tgt_lang
        )
        return translated
    except Exception as e:
        return f"Translation failed: {str(e)}"

def pipeline(audio_path: str, target_language: str) -> tuple:
    """Full processing pipeline"""
    lang, text = transcribe_and_detect_language(audio_path)
    translated = translate(lang, text, target_language)
    return lang, text, translated

# Create Gradio interface
with gr.Blocks(title="Multilingual Audio Translator") as demo:
    gr.Markdown("## 🎙️ Multilingual Audio Translator")
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"],
            type="filepath",
            label="Upload or Record Audio"
        )
        
    with gr.Row():
        language_dropdown = gr.Dropdown(
            choices=sorted(list(LANGUAGES.values())),
            label="Translate To",
            value="English"
        )
    
    with gr.Row():
        lang_out = gr.Textbox(label="Detected Language")
    with gr.Row():
        transcript_out = gr.Textbox(label="Transcribed Text", lines=4)
    with gr.Row():
        translation_out = gr.Textbox(label="Translated Text", lines=4)
    
    translate_btn = gr.Button("Translate", variant="primary")
    
    translate_btn.click(
        fn=pipeline,
        inputs=[audio_input, language_dropdown],
        outputs=[lang_out, transcript_out, translation_out]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
