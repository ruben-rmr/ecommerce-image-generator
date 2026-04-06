import requests

PHOTOROOM_API_KEY = "sandbox_sk_pr_default_5f041be8654a918c51a598aaa85efb66f9afdd87"
PHOTOROOM_URL = "https://image-api.photoroom.com/v2/edit"

def generate_background_via_api(image_bytes: bytes, prompt: str) -> bytes:
    """
    Envía la imagen segmentada a Photoroom y devuelve los bytes de la nueva imagen.
    """
    headers = {
        "x-api-key": PHOTOROOM_API_KEY
    }
    
    data = {
        "background.prompt": prompt,
        "padding": "0.1", # Margen para que el objeto respire
    }
    
    files = {
        "imageFile": ("objeto_limpio.png", image_bytes, "image/png")
    }

    response = requests.post(PHOTOROOM_URL, headers=headers, data=data, files=files)

    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Photoroom API Error ({response.status_code}): {response.text}")