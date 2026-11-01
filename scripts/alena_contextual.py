#!/usr/bin/env python3
"""Belarusian context-sensitive lexical-stress adapter for an eSpeak-trained Piper voice.

Preserves eSpeak phones and all punctuation, changes only primary stress U+02C8
on confidently aligned tokens. The code DOES NOT magically disambiguate unknown
homographs; it reports them. Input wordform lexicon is supplied locally by user.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import unicodedata
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

WORD_PATTERN = re.compile(r"[А-Яа-яЁёІіЎўҐґ]+(?:[-’'][А-Яа-яЁёІіЎўҐґ]+)*")
IPA_WORD_PATTERN = re.compile(r"\S+")
TEXT_VOWELS = frozenset("аеёіоуыэюя")
IPA_VOWELS = frozenset("aeiouyɛɨɐɑæɒɛəɜɝɚɪɔøœʊʉɯɤʌɵɘɞɶʏ")
STRESS = "ˈ"
TIE_MARK = "\u0361"  # known missing character in original Alena phoneme map
SPLIT_SENTENCE = re.compile(r"(?<=[.!?…])\s+")


@dataclass
class Decision:
    word: str
    status: str
    source: str
    chosen_vowel: int | None
    options: list[int]
    original_ipa: str = ""
    corrected_ipa: str = ""


@dataclass
class Outcome:
    text: str
    ipa: str
    decisions: list[Decision]
    unresolved: list[Decision]


class Lexicon:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"Не найден словарь: {path}")
        opener = gzip.open if str(path).endswith(".gz") else open
        self.entries: dict[str, tuple[int, ...]] = {}
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                word, sep, indices = line.rstrip("\r\n").partition("\t")
                if not sep:
                    continue
                try:
                    candidates = tuple(sorted({int(v) for v in indices.split(",") if v.strip()}))
                    if candidates:
                        self.entries[word.casefold()] = candidates
                except ValueError:
                    continue
        if len(self.entries) < 1:
            raise ValueError("Словарь ударений пуст")

    def lookup(self, word: str) -> tuple[int, ...]:
        word = word.casefold()
        opts = self.entries.get(word)
        if opts is None and word.startswith("ў"):
            fallback = self.entries.get("у" + word[1:])
            if fallback:
                opts = tuple(v - 1 for v in fallback if v > 1)
        n = len(vowels(word))
        return tuple(i for i in (opts or ()) if 1 <= i <= n)


def vowels(word: str) -> list[int]:
    return [i for i, c in enumerate(word.casefold()) if c in TEXT_VOWELS]


class Rules:
    def __init__(self, path: Path):
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.word_rules: dict[str, dict[str, Any]] = {}
        self.context_rules: list[tuple[str, re.Pattern, int, str]] = []
        for word, item in raw.get("word_rules", {}).items():
            value = int(item["vowel"])
            if value <= 0 or value > len(vowels(word)):
                raise ValueError(f"Неверная гласная в word_rules: {word} → {value}")
            if not item.get("source"):
                raise ValueError(f"Нет источника для словарного исключения: {word}")
            self.word_rules[word.casefold()] = item
        for item in raw.get("context_rules", []):
            word = item["word"].casefold()
            value = int(item["vowel"])
            if value <= 0 or value > len(vowels(word)):
                raise ValueError(f"Неверная гласная в context_rules: {word} → {value}")
            regex = re.compile(item["regex"], re.IGNORECASE | re.UNICODE)
            source = item.get("source", "")
            if not source:
                raise ValueError(f"Нет источника для контекстного исключения: {word}")
            self.context_rules.append((word, regex, value, source))

    def context_choice(self, text: str, word: str, start: int, end: int) -> tuple[int, str] | None:
        for target, regex, vowel, source in self.context_rules:
            if word.casefold() != target:
                continue
            for match in regex.finditer(text):
                if match.start() <= start and end <= match.end():
                    return vowel, source
        return None


def choose(word: str, text: str, start: int, end: int, lexicon: Lexicon,
           rules: Rules) -> Decision:
    key = word.casefold()
    v = vowels(key)
    opts = list(lexicon.lookup(key))
    if len(v) <= 1:
        return Decision(word, "single_syllable", "", None, opts)
    ctx = rules.context_choice(text, key, start, end)
    if ctx:
        chosen, source = ctx
        return Decision(word, "context_rule", source, chosen, opts)
    custom = rules.word_rules.get(key)
    if custom:
        return Decision(word, "word_rule", custom["source"], int(custom["vowel"]), opts)
    if "ё" in key:
        chosen = next(i for i, idx in enumerate(v, 1) if key[idx] == "ё")
        return Decision(word, "yo_rule", "orthographic ё", chosen, opts)
    if len(opts) == 1:
        return Decision(word, "dictionary_unique", "GrammarDB-derived local lexicon", opts[0], opts)
    if len(opts) > 1:
        return Decision(word, "ambiguous", "several dictionary readings", None, opts)
    return Decision(word, "not_in_lexicon", "word not found in dictionary", None, opts)


def move_stress(part: str, decision: Decision) -> Decision:
    decision.original_ipa = part
    decision.corrected_ipa = part
    if decision.chosen_vowel is None:
        return decision
    if part.count(STRESS) > 1:
        decision.status = "multiple_primary_stresses"
        return decision
    # We do not change non-stress IPA symbols.
    bare = part.replace(STRESS, "")
    ipav = [i for i, c in enumerate(bare) if c in IPA_VOWELS]
    if len(ipav) != len(vowels(decision.word)):
        decision.status = "vowel_alignment_mismatch"
        return decision
    if decision.chosen_vowel > len(ipav):
        decision.status = "vowel_index_out_of_range"
        return decision
    target = ipav[decision.chosen_vowel - 1]
    decision.corrected_ipa = bare[:target] + STRESS + bare[target:]
    return decision


def normalize_inventory(ipa: str, valid_map: dict[str, Any]) -> str:
    # Trained Alena skips this tie mark; do so deterministically both here and in synthesis.
    clean = ipa.replace(TIE_MARK, "")
    chars = set(unicodedata.normalize("NFD", clean))
    not_in_map = chars.difference(valid_map)
    if not_in_map:
        raise ValueError(f"IPA содержит отсутствующие в карте модели символы: {sorted(not_in_map)}")
    return clean


def process_one_sentence(text: str, phones: list[str], lexicon: Lexicon,
                         rules: Rules, valid_map: dict[str, Any]) -> Outcome:
    original = "".join(phones)
    source_tokens = list(WORD_PATTERN.finditer(text))
    ipa_tokens = list(IPA_WORD_PATTERN.finditer(original))
    if len(source_tokens) != len(ipa_tokens):
        raise ValueError(
            f"Нельзя надёжно выровнять текст и IPA: {len(source_tokens)} слов, "
            f"{len(ipa_tokens)} фонетических групп; не делаем догадок. "
            f"Текст: {text!r}, IPA: {original!r}"
        )
    decisions: list[Decision] = []
    fragments: list[str] = []
    cursor = 0
    for source, phone in zip(source_tokens, ipa_tokens):
        decision = choose(source.group(), text, source.start(), source.end(), lexicon, rules)
        move_stress(phone.group(), decision)
        decisions.append(decision)
        fragments.append(original[cursor:phone.start()])
        fragments.append(decision.corrected_ipa)
        cursor = phone.end()
    fragments.append(original[cursor:])
    final = normalize_inventory("".join(fragments), valid_map)
    problematic = [d for d in decisions if d.status in {
        "ambiguous", "not_in_lexicon", "multiple_primary_stresses",
        "vowel_alignment_mismatch", "vowel_index_out_of_range"
    }]
    return Outcome(text, final, decisions, problematic)


def process_text(text: str, voice: Any, lexicon: Lexicon, rules: Rules,
                 valid_map: dict[str, Any]) -> list[Outcome]:
    if "[[" in text or "]]" in text:
        raise ValueError("Входной текст не должен содержать raw IPA блоки [[ ... ]]")
    sentences = [s.strip() for s in SPLIT_SENTENCE.split(text) if s.strip()]
    results: list[Outcome] = []
    for sentence in sentences:
        all_phones = voice.phonemize(sentence)
        if len(all_phones) != 1:
            raise ValueError(
                f"Piper разделил предложение на {len(all_phones)} частей; "
                "нужно разбить входные фразы вручную: " + repr(sentence)
            )
        results.append(process_one_sentence(sentence, all_phones[0], lexicon, rules, valid_map))
    return results


def phonemes_from_raw_block(voice: Any, ipa: str) -> None:
    check = voice.phonemize(f"[[{ipa}]]")
    if len(check) != 1 or "".join(check[0]).strip() != ipa.strip():
        raise RuntimeError("Piper изменил IPA внутри [[...]], синтез отменён")


def print_outcome(outcome: Outcome) -> None:
    print("\nТекст:", outcome.text)
    print("IPA:  ", outcome.ipa)
    print("Изменения и контекстные правила:")
    for d in outcome.decisions:
        if d.chosen_vowel is not None and d.original_ipa != d.corrected_ipa:
            print(f"  {d.word}: {d.original_ipa} -> {d.corrected_ipa} [{d.status}]")
    if outcome.unresolved:
        print("Неразрешённые ударения/сопоставления:")
        for d in outcome.unresolved:
            print(f"  {d.word}: {d.status}, варианты: {d.options}, IPA: {d.original_ipa}")
    else:
        print("Нет неразрешённых словоформ среди многосложных слов.")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "synthesize"))
    parser.add_argument("--text", required=True, help="Белорусский текст")
    parser.add_argument("--model", default=str(root / "models" / "alena_stress_epoch859.onnx"))
    parser.add_argument("--lexicon", default=str(root / "resources" / "be-stress.tsv.gz"))
    parser.add_argument("--rules", default=str(root / "scripts" / "config" / "alena_context_rules.json"))
    parser.add_argument("--strict", action="store_true", help="Отклонять неразрешённые омографы/неизвестные слова")
    parser.add_argument("--output", default=str(root / "output.wav"))
    parser.add_argument("--audit", default="", help="Путь для JSON аудита обработки")
    parser.add_argument("--length-scale", type=float, default=1.0)
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("Не передан текст")
    if args.length_scale <= 0:
        parser.error("length-scale должен быть больше нуля")

    from piper import PiperVoice, SynthesisConfig

    model_path = Path(args.model)
    cfg_path = Path(str(model_path) + ".json")
    if not model_path.is_file() or not cfg_path.is_file():
        raise FileNotFoundError(f"Требуются ONNX и JSON рядом с ней: {model_path}, {cfg_path}")
    model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if model_cfg.get("phoneme_type") != "espeak" or model_cfg.get("espeak", {}).get("voice") not in ("be", None):
        raise ValueError("Модель должна иметь совместимую исходную конфигурацию eSpeak 'be'")
    inventory = model_cfg["phoneme_id_map"]
    voice = PiperVoice.load(str(model_path))
    lexicon = Lexicon(Path(args.lexicon))
    rules = Rules(Path(args.rules))
    outcomes = process_text(args.text, voice, lexicon, rules, inventory)
    for result in outcomes:
        print_outcome(result)

    unresolved = [d for result in outcomes for d in result.unresolved]
    if args.audit:
        audit = Path(args.audit)
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps({
            "model": str(model_path), "original_text": args.text,
            "sentences": [
                {"text": r.text, "ipa": r.ipa, "decisions": [asdict(d) for d in r.decisions]}
                for r in outcomes
            ],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Аудит:", audit)

    if unresolved and args.strict:
        print(f"STRICT: {len(unresolved)} неразрешённых случаев; синтез отменён", file=sys.stderr)
        return 2
    if args.command == "inspect":
        return 0

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    settings = SynthesisConfig(length_scale=args.length_scale, noise_scale=0.667, noise_w_scale=0.8)
    # Each sentence is synthesized separately, allowing the model to handle punctuation/end stops.
    # Model WAV metadata is identical for all sentences.
    sample_rate = int(model_cfg["audio"]["sample_rate"])
    with wave.open(str(outpath), "wb") as final_wav:
        final_wav.setnchannels(1)
        final_wav.setsampwidth(2)
        final_wav.setframerate(sample_rate)
        for result in outcomes:
            phonemes_from_raw_block(voice, result.ipa)
            # Avoid nested wave context; synthesize_wav streams PCM to this wave writer.
            voice.synthesize_wav(
                f"[[{result.ipa}]]", final_wav, syn_config=settings, set_wav_format=False
            )
    print("WAV сохранён:", outpath)
    if unresolved:
        print("Внимание: сохраняются неразрешённые слова. Не заявляйте о 100%-корректном ударении.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ОШИБКА:", exc, file=sys.stderr)
        sys.exit(1)
