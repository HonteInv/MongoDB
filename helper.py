"""
helper.py — shared utilities for the Portfolio AI system.
"""

import os


def get_latest_opus_model(client, fallback: str = "claude-opus-5") -> str:
    """
    Resolve which Claude model to use.

    1. If CLAUDE_MODEL is set in the environment, that wins — return it directly.
       (Set CLAUDE_MODEL=claude-sonnet-4-6 in .env to test cheaply, etc.)
    2. Otherwise query the Anthropic API and return the most recent claude-opus-*
       model available on this API key (returned newest-first, so index 0).
    3. If the API call fails or returns no Opus models, return `fallback`.

    Usage:
        from anthropic import Anthropic
        from helper import get_latest_opus_model

        client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        model  = get_latest_opus_model(client)
    """
    # 1. Explicit override always wins — no API call needed.
    override = os.getenv("CLAUDE_MODEL")
    if override:
        print(f"  [helper] Using CLAUDE_MODEL override: {override}")
        return override

    try:
        # Iterate the page object directly — it auto-paginates across all pages.
        # (`models.data` is only the FIRST page and could miss newer models.)
        opus_models = [m.id for m in client.models.list() if "opus" in m.id]
        if opus_models:
            latest = opus_models[0]
            print(f"  [helper] Latest Opus model resolved: {latest}")
            return latest
    except Exception as e:
        print(f"  [helper] Could not fetch model list ({e}) — using fallback: {fallback}")
    return fallback
