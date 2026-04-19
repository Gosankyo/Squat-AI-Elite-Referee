import speech_recognition as sr
import time

class NLPVoiceAssistant:
    """
    Automatic Speech Recognition (ASR) module for Squat AI.
    Listens for specific voice commands to trigger the Computer Vision pipeline,
    satisfying the Speech-to-Text requirements for the NLP architecture.
    """
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Adjust energy threshold for gym environments (background noise)
        self.recognizer.energy_threshold = 400 
        
    def wait_for_command(self, wake_word="start analysis", language="en-US"):
        """
        Continuously listens to the microphone until the wake_word is detected.
        Uses Google's Web Speech API for transcription.
        """
        print(f"\n[NLP Module] Microphone active. Waiting for voice command: '{wake_word}'...")
        
        with sr.Microphone() as source:
            # Calibrate for ambient noise for 1 second
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while True:
                try:
                    # Listen for a phrase (max 5 seconds per phrase)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Transcribe speech to text
                    text = self.recognizer.recognize_google(audio, language=language).lower()
                    print(f"  -> [Speech-to-Text] Transcribed: '{text}'")
                    
                    if wake_word in text:
                        print(f"\n[SUCCESS] [NLP Module] Command '{wake_word}' recognized! Triggering Pipeline...")
                        return True
                    elif "stop" in text or "exit" in text:
                        print("\n[STOP] [NLP Module] Stop command recognized. Aborting.")
                        return False
                        
                except sr.WaitTimeoutError:
                    # No speech detected, continue looping
                    continue
                except sr.UnknownValueError:
                    # Audio detected but could not be transcribed
                    continue
                except sr.RequestError as e:
                    print(f"[ERROR] [NLP Module] API unavailable: {e}")
                    return False

# --- Testing the module independently ---
if __name__ == "__main__":
    assistant = NLPVoiceAssistant()
    assistant.wait_for_command(wake_word="start analysis", language="en-US")