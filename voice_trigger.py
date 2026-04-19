import speech_recognition as sr
import time

class NLPVoiceAssistant:
    """
    Automatic Speech Recognition (ASR) module.
    Implements voice command detection for pipeline triggering.
    Uses Google Speech-to-Text API for command transcription.
    """
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Energy threshold set for high-noise environments (400 dB)
        self.recognizer.energy_threshold = 400 
        
    def wait_for_command(self, wake_word="start analysis", language="en-US"):
        """
        Listens to microphone input until wake word is detected.
        Returns True on successful command recognition, False on abort.
        """
        print(f"[ASR] Listening for command: '{wake_word}'")
        
        with sr.Microphone() as source:
            # Calibrate ambient noise baseline (1 second)
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while True:
                try:
                    # Listen for audio input (5 second limit, 1 second timeout)
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                    # Send to Google STT API for transcription
                    text = self.recognizer.recognize_google(audio, language=language).lower()
                    print(f"[ASR] Recognized: '{text}'")
                    
                    if wake_word in text:
                        print(f"[ASR] Command matched. Pipeline initialization starting...")
                        return True
                    elif "stop" in text or "exit" in text:
                        print("[ASR] Abort command received. Shutting down.")
                        return False
                        
                except sr.WaitTimeoutError:
                    # Timeout waiting for speech. Continue listening.
                    continue
                except sr.UnknownValueError:
                    # Audio present but unintelligible. Continue listening.
                    continue
                except sr.RequestError as e:
                    print(f"[ERROR] API connection failed: {e}")
                    return False

# --- Testing the module independently ---
if __name__ == "__main__":
    assistant = NLPVoiceAssistant()
    assistant.wait_for_command(wake_word="start analysis", language="en-US")