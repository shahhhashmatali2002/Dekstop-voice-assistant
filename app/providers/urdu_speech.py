"""Making Urdu text pronounceable, and making clauses sound joined up.

Spec reference: §8.3, §8.4.3.

Two complaints about the spoken output - "some words are not pronounced
properly" and "it sounds like a robot" - have two different causes, and
neither of them is the voice.

WHY WORDS COME OUT WRONG
------------------------
The translator is deliberately told to carry technical terms, product names
and people's names through as they are (§8.4.3, rule 7). That is right for
the transcript: a reader wants to see "deployment", not a guess at how to
spell it in Urdu. It is wrong for a voice. A Pakistani neural voice handed
Latin letters inside an Urdu sentence does the only thing it can - it guesses,
and the guess is either an English reading with an Urdu accent laid over it or
a letter-by-letter spelling. "API" becomes something between "api" and "a-p-i";
"YouTube" comes out as neither.

A human interpreter does not do that. They say the word the way an Urdu
speaker says it - "ڈیپلائمنٹ", "یوٹیوب", "اے پی آئی" - so that is what the
voice is given here, at the last possible moment, after the transcript has
already been written with the original spelling.

The dictionary is explicit rather than algorithmic. Rule-based Latin-to-Urdu
transliteration produces confident nonsense for anything it has not seen, and
a wrong Urdu spelling is pronounced wrongly with total conviction, which is
worse than the voice's own hedge. So a word we know is converted, an acronym
is spelled out with Urdu letter names, and anything else is left alone for the
voice to do its best with. Adding a word to `WORDS` is a one-line fix and is
meant to be.

WHY IT SOUNDS LIKE A ROBOT
--------------------------
That is punctuation, not the voice. A streaming recogniser hands over clauses,
not sentences, and every clause used to be closed with a full stop before
synthesis - because a fragment with no final mark is read flat, so something
had to be added. A voice reading "۔" does what the mark means: it drops its
pitch and stops. One sentence delivered as four clauses was therefore read as
four separate statements, each falling to a close. Four full stops where the
speaker had none is exactly what a machine reading a list sounds like.

A fragment now gets "،" instead - a continuation: a short pause, no drop in
pitch, the sentence audibly unfinished. Only a clause that really ends an
utterance, because the recogniser said so or because the speaker stopped
talking, is closed with "۔".
"""

from __future__ import annotations

import re

#: Marks that make a voice fall and stop.
TERMINAL = "۔؟!…"

#: Marks that make it pause without closing the thought.
CONTINUATION = "،؛:,;"

#: A line already ending in any of these needs nothing added. The ASCII marks
#: are in here because a translation frequently comes back with an English
#: full stop or question mark, and appending an Urdu one after it - "بات.۔" -
#: is read as two endings: a stop, a pause, and another stop.
CLOSED = TERMINAL + CONTINUATION + ".?"

#: How an Urdu speaker says the English words that survive translation.
#: Keys are lower-case; lookups strip surrounding punctuation and hyphens.
WORDS: dict[str, str] = {
    # everyday
    "hello": "ہیلو", "hi": "ہائے", "hey": "ہے", "bye": "بائے",
    "okay": "اوکے", "ok": "اوکے", "yes": "یس", "no": "نو",
    "please": "پلیز", "thanks": "تھینکس", "thank": "تھینک",
    "sorry": "سوری", "welcome": "ویلکم", "sir": "سر", "madam": "میڈم",
    "sure": "شور", "fine": "فائن", "good": "گڈ", "great": "گریٹ",
    # meetings and business
    "meeting": "میٹنگ", "call": "کال", "team": "ٹیم", "office": "آفس",
    "company": "کمپنی", "business": "بزنس", "client": "کلائنٹ",
    "customer": "کسٹمر", "manager": "مینیجر", "project": "پروجیکٹ",
    "report": "رپورٹ", "update": "اپڈیٹ", "email": "ای میل",
    "message": "میسج", "schedule": "شیڈول", "deadline": "ڈیڈ لائن",
    "budget": "بجٹ", "invoice": "انوائس", "payment": "پیمنٹ",
    "contract": "کنٹریکٹ", "agreement": "ایگریمنٹ", "policy": "پالیسی",
    "plan": "پلان", "strategy": "اسٹریٹجی", "target": "ٹارگٹ",
    "goal": "گول", "sales": "سیلز", "market": "مارکیٹ",
    "revenue": "ریونیو", "profit": "پرافٹ", "cost": "کاسٹ",
    "price": "پرائس", "order": "آرڈر", "delivery": "ڈیلیوری",
    "interview": "انٹرویو", "salary": "سیلری", "process": "پروسیس",
    "presentation": "پریزنٹیشن", "document": "ڈاکومنٹ",
    "documentation": "ڈاکومنٹیشن", "feedback": "فیڈ بیک",
    "review": "ریویو", "support": "سپورٹ", "service": "سروس",
    "product": "پروڈکٹ", "quality": "کوالٹی", "level": "لیول",
    # software and hardware
    "server": "سرور", "system": "سسٹم", "software": "سافٹ ویئر",
    "hardware": "ہارڈ ویئر", "computer": "کمپیوٹر", "laptop": "لیپ ٹاپ",
    "mobile": "موبائل", "screen": "اسکرین", "camera": "کیمرہ",
    "microphone": "مائیکروفون", "speaker": "اسپیکر", "audio": "آڈیو",
    "video": "ویڈیو", "internet": "انٹرنیٹ", "network": "نیٹ ورک",
    "website": "ویب سائٹ", "browser": "براؤزر", "link": "لنک",
    "file": "فائل", "folder": "فولڈر", "code": "کوڈ", "data": "ڈیٹا",
    "database": "ڈیٹا بیس", "backup": "بیک اپ", "cloud": "کلاؤڈ",
    "download": "ڈاؤن لوڈ", "upload": "اپ لوڈ", "install": "انسٹال",
    "setup": "سیٹ اپ", "settings": "سیٹنگز", "application": "ایپلی کیشن",
    "app": "ایپ", "feature": "فیچر", "bug": "بگ", "fix": "فکس",
    "issue": "ایشو", "ticket": "ٹکٹ", "sprint": "اسپرنٹ",
    "release": "ریلیز", "version": "ورژن", "deployment": "ڈیپلائمنٹ",
    "deploy": "ڈیپلائے", "production": "پروڈکشن", "staging": "اسٹیجنگ",
    "development": "ڈیولپمنٹ", "developer": "ڈیولپر", "design": "ڈیزائن",
    "test": "ٹیسٹ", "testing": "ٹیسٹنگ", "security": "سیکیورٹی",
    "password": "پاس ورڈ", "account": "اکاؤنٹ", "login": "لاگ ان",
    "performance": "پرفارمنس", "latency": "لیٹنسی", "model": "ماڈل",
    "training": "ٹریننگ", "machine": "مشین", "learning": "لرننگ",
    "translation": "ٹرانسلیشن", "language": "لینگویج", "voice": "وائس",
    # learning and video
    "channel": "چینل", "subscribe": "سبسکرائب", "comment": "کمنٹ",
    "share": "شیئر", "follow": "فالو", "class": "کلاس",
    "course": "کورس", "lesson": "لیسن", "student": "اسٹوڈنٹ",
    "teacher": "ٹیچر", "exam": "ایگزام", "practice": "پریکٹس",
    "beginner": "بگنر", "intermediate": "انٹرمیڈیٹ",
    "advanced": "ایڈوانسڈ", "english": "انگلش", "urdu": "اردو",
    # names people actually say
    "youtube": "یوٹیوب", "google": "گوگل", "microsoft": "مائیکروسافٹ",
    "teams": "ٹیمز", "zoom": "زوم", "windows": "ونڈوز",
    "chrome": "کروم", "facebook": "فیس بک", "whatsapp": "واٹس ایپ",
    "instagram": "انسٹاگرام", "twitter": "ٹویٹر", "linkedin": "لنکڈ ان",
    "amazon": "ایمازون", "apple": "ایپل", "android": "اینڈرائیڈ",
    "iphone": "آئی فون", "gmail": "جی میل", "excel": "ایکسل",
    "powerpoint": "پاور پوائنٹ", "outlook": "آؤٹ لک",
    "openai": "اوپن اے آئی", "chatgpt": "چیٹ جی پی ٹی",
    "github": "گٹ ہب", "python": "پائیتھن", "javascript": "جاوا اسکرپٹ",
    "ielts": "آئلٹس", "wifi": "وائی فائی",
}

#: Urdu names for the Latin letters, for acronyms that must be spelled.
LETTER_NAMES = {
    "a": "اے", "b": "بی", "c": "سی", "d": "ڈی", "e": "ای", "f": "ایف",
    "g": "جی", "h": "ایچ", "i": "آئی", "j": "جے", "k": "کے", "l": "ایل",
    "m": "ایم", "n": "این", "o": "او", "p": "پی", "q": "کیو", "r": "آر",
    "s": "ایس", "t": "ٹی", "u": "یو", "v": "وی", "w": "ڈبلیو",
    "x": "ایکس", "y": "وائی", "z": "زیڈ",
}

#: Loanword plurals: "servers" is said "سرورز", not "سرور".
_PLURAL_SUFFIX = "ز"

_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z.\-'’]*[A-Za-z]|[A-Za-z]")
_PERCENT = re.compile(r"(\d)\s*%")
_DOLLARS = re.compile(r"\$\s*(\d[\d,.]*)")
_POUNDS = re.compile(r"£\s*(\d[\d,.]*)")
_EUROS = re.compile(r"€\s*(\d[\d,.]*)")


def spell_acronym(token: str) -> str:
    """"API" -> "اے پی آئی". Letters we have no name for are left alone."""
    letters = [LETTER_NAMES.get(ch.lower(), ch) for ch in token if ch.isalpha()]
    return " ".join(letters)


def say_word(token: str) -> str:
    """One Latin token as an Urdu speaker would say it.

    Returns the token unchanged when we have no better idea than the voice
    does - which is the honest answer more often than a dictionary this size
    would like to admit.
    """
    core = token.strip(".-'’")
    if not core:
        return token
    key = core.lower().replace("-", "").replace(".", "").replace("'", "")

    known = WORDS.get(key)
    if known:
        return known

    # A loanword plural: "servers", "tickets", "files".
    if key.endswith("s") and len(key) > 3:
        singular = WORDS.get(key[:-1])
        if singular:
            return singular + _PLURAL_SUFFIX

    # An acronym, whether or not it was written with dots. "A.P.I." and "API"
    # are the same word said aloud.
    stripped = core.replace(".", "")
    if stripped.isupper() and 2 <= len(stripped) <= 6:
        return spell_acronym(stripped)

    # A single letter is always a letter: "plan B", "vitamin D".
    if len(stripped) == 1 and stripped.isalpha():
        return LETTER_NAMES.get(stripped.lower(), stripped)

    return token


def _symbols(text: str) -> str:
    """Symbols a voice reads in the wrong language, or in the wrong order.

    Currency is the reason this is patterns rather than a character map: "$5"
    is said "پانچ ڈالر", so the symbol has to move to the other side of the
    number, not simply be replaced where it stands.
    """
    text = _PERCENT.sub(r"\1 فیصد", text)
    text = _DOLLARS.sub(r"\1 ڈالر", text)
    text = _POUNDS.sub(r"\1 پاؤنڈ", text)
    text = _EUROS.sub(r"\1 یورو", text)
    return text


def pronounceable(text: str) -> str:
    """Rewrite the parts of an Urdu line that a Urdu voice cannot read.

    Only Latin runs and a few symbols are touched. Urdu text, digits and
    punctuation are left exactly as they are: the voice reads those correctly
    already, and every unnecessary edit is a chance to make it worse.
    """
    if not text:
        return ""
    out = _symbols(text)
    out = _LATIN_RUN.sub(lambda m: say_word(m.group(0)), out)
    return " ".join(out.split())


def close(text: str, *, final: bool = True) -> str:
    """Give a clause an ending that matches whether the thought is finished.

    `final` should be true only when the utterance really ended - the
    recogniser marked the hypothesis final, or the speaker stopped. A mid
    sentence clause closed with a full stop is the robotic reading this whole
    module exists to stop.
    """
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    if clean[-1] in CLOSED:
        return clean
    return clean + ("۔" if final else "،")


def ends_sentence(source_text: str, reason: str = "") -> bool:
    """Did this unit end a sentence, judged from the English it came from?

    The English carries recogniser punctuation; the Urdu often does not,
    because the translator is told not to invent an ending the speaker did not
    imply. So finality is read off the source, not the translation.
    """
    if reason in ("final", "vad"):
        return True
    stripped = (source_text or "").strip()
    return bool(stripped) and stripped[-1] in ".!?۔؟…"


def for_speech(text: str, *, final: bool = True) -> str:
    """Everything above, in the order it has to happen."""
    return close(pronounceable(text), final=final)
