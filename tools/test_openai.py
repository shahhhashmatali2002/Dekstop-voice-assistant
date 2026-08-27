#!/usr/bin/env python
"""Test OpenAI API key validity, latency, model access, and quota.

Usage:
    python tools/test_openai.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.models.config import Config, get_secret
from app.providers.llm_translator import LlmTranslator
from app.providers.base import TranslationUnit, RollingContext


def _utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


async def test_openai() -> int:
    _utf8_console()
    cfg = Config.load()
    key = get_secret("OPENAI_API_KEY")
    model = cfg.providers.translator_model or "gpt-4o-mini"
    
    print("\n==================================================")
    print("  OPENAI API KEY & TRANSLATION TEST")
    print("==================================================")
    
    if not key:
        print("\n  [FAIL] OPENAI_API_KEY is not set.")
        print("         Add it to your .env file or Windows Credential Manager.")
        return 1
        
    masked = key[:7] + "..." + key[-4:] if len(key) > 12 else "..."
    print(f"\n  • API Key Present:  {masked}")
    print(f"  • Model Configured: {model}")
    
    translator = LlmTranslator(api_key=key, model=model, timeout_s=10.0)
    unit = TranslationUnit(
        text="Hello, this is a live test of the OpenAI meeting translator.",
        source_language="en",
        target_language="ur",
    )
    
    print("\n  Connecting to OpenAI API...")
    t0 = time.time()
    try:
        translated = await translator.translate(unit, RollingContext())
        elapsed = round((time.time() - t0) * 1000, 1)
        
        print("\n  [OK] SUCCESS: OpenAI API key is VALID and ACTIVE!")
        print(f"  • Response Latency: {elapsed} ms")
        print(f"  • English Input:    {unit.text}")
        print(f"  • Urdu Translation: {translated.text}")
        print("\n==================================================\n")
        return 0
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 1)
        err_msg = str(exc)
        print(f"\n  [FAIL] OpenAI Request Failed after {elapsed} ms")
        print(f"  • Error Details: {err_msg}")
        
        if "401" in err_msg or "invalid_api_key" in err_msg.lower() or "incorrect api key" in err_msg.lower():
            print("\n  DIAGNOSIS: INVALID API KEY")
            print("  Your OPENAI_API_KEY was rejected by OpenAI. Please check for typos or regenerate your key.")
        elif "429" in err_msg or "quota" in err_msg.lower() or "insufficient_quota" in err_msg.lower():
            print("\n  DIAGNOSIS: QUOTA EXCEEDED / OUT OF CREDITS")
            print("  You have hit your OpenAI API rate limit or run out of billing credits.")
            print("  Check your usage at https://platform.openai.com/usage")
        elif "404" in err_msg or "model_not_found" in err_msg.lower():
            print(f"\n  DIAGNOSIS: MODEL NOT FOUND ({model})")
            print("  The requested model is not accessible with this key.")
        else:
            print("\n  DIAGNOSIS: Network/Connection Issue or Timeout")
            
        print("\n==================================================\n")
        return 1



if __name__ == "__main__":
    raise SystemExit(asyncio.run(test_openai()))
