"""
helper.py — shared utilities for the Portfolio AI system.
"""

import os


def get_latest_opus_model(client, fallback: str = "claude-opus-4-7") -> str:
    """
    Query the Anthropic API and return the most recent claude-opus-* model
    available on this API key.

    Models are returned newest-first by the API, so index 0 is always the
    latest.  Falls back to `fallback` if the API call fails or returns no
    Opus models.

    Usage:
        from anthropic import Anthropic
        from helper import get_latest_opus_model

        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        model  = get_latest_opus_model(client)
    """
    try:
        models = client.models.list()
        opus_models = [m.id for m in models.data if "opus" in m.id]
        if opus_models:
            latest = opus_models[0]
            print(f"  [helper] Latest Opus model resolved: {latest}")
            return latest
    except Exception as e:
        print(f"  [helper] Could not fetch model list ({e}) — using fallback: {fallback}")
    return fallback
