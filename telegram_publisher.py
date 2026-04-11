import os
import requests
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

API_URL = os.environ["API_URL_TO_POST"]
API_KEY = os.environ["API_SECRET_KEY"]


def post_to_telegram(channel_id: str, text: str, photo: str | None = None) -> dict:
    payload = {"channel_id": channel_id, "text": text}
    if photo:
        payload["photo"] = photo

    response = requests.post(
        API_URL,
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        json=payload,
    )

    result = response.json()
    if result.get("status") != "ok":
        raise RuntimeError(f"Post failed: {result.get('detail')}")

    return result