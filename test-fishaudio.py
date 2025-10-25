import requests
import pygame
import tempfile
import os

def speak(text):
    response = requests.post(
        "https://api.fish.audio/v1/tts",
        headers={
            "Authorization": f"Bearer 04a6fd56da0746feb683c18b84ab99b1",
            "Content-Type": "application/json",
            "model": "s1"
        },
        json={"text": text, "format": "mp3"}
    )
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        f.write(response.content)
        temp_path = f.name
    
    pygame.mixer.init()
    pygame.mixer.music.load(temp_path)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    
    os.unlink(temp_path)

if __name__ == "__main__":
    speak("Hello! This is a test.")
