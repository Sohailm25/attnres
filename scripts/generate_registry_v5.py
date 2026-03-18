#!/usr/bin/env python
# ABOUTME: Deterministically generates the stratified oracle prompt registry_v5 from the saved v4 base registry.
# ABOUTME: Expands only the oracle-alpha collection while preserving the non-oracle collections unchanged.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_REGISTRY_PATH = ROOT / "prompts" / "registry_v4.yaml"
DEFAULT_OUTPUT_PATH = ROOT / "prompts" / "registry_v5.yaml"
PILOT_RECORDS_PER_SUBCATEGORY = 4
CONFIRM_RECORDS_PER_SUBCATEGORY = 16


@dataclass(frozen=True)
class TemplatePair:
    text: str
    paraphrase: str


@dataclass(frozen=True)
class SubcategorySpec:
    name: str
    records: tuple[dict[str, Any], ...]
    template_pairs: tuple[TemplatePair, ...]


def _indefinite_article(noun_phrase: str) -> str:
    return "an" if noun_phrase[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def _split_records(
    records: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    expected = PILOT_RECORDS_PER_SUBCATEGORY + CONFIRM_RECORDS_PER_SUBCATEGORY
    if len(records) != expected:
        raise ValueError(f"expected {expected} records, found {len(records)}")
    return records[:PILOT_RECORDS_PER_SUBCATEGORY], records[
        PILOT_RECORDS_PER_SUBCATEGORY:
    ]


def _capital_records() -> tuple[dict[str, Any], ...]:
    facts = (
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Brazil", "Brasilia"),
        ("Kenya", "Nairobi"),
        ("Canada", "Ottawa"),
        ("Spain", "Madrid"),
        ("Australia", "Canberra"),
        ("Egypt", "Cairo"),
        ("Norway", "Oslo"),
        ("Argentina", "Buenos Aires"),
        ("Thailand", "Bangkok"),
        ("Peru", "Lima"),
        ("Poland", "Warsaw"),
        ("Ethiopia", "Addis Ababa"),
        ("New Zealand", "Wellington"),
        ("Vietnam", "Hanoi"),
        ("Portugal", "Lisbon"),
        ("South Korea", "Seoul"),
        ("Chile", "Santiago"),
        ("Turkey", "Ankara"),
    )
    return tuple({"country": country, "city": city} for country, city in facts)


def _author_records() -> tuple[dict[str, Any], ...]:
    facts = (
        ("Pride and Prejudice", "Jane Austen"),
        ("1984", "George Orwell"),
        ("Beloved", "Toni Morrison"),
        ("The Trial", "Franz Kafka"),
        ("Moby-Dick", "Herman Melville"),
        ("Invisible Man", "Ralph Ellison"),
        ("Frankenstein", "Mary Shelley"),
        ("The Odyssey", "Homer"),
        ("Jane Eyre", "Charlotte Bronte"),
        ("The Great Gatsby", "F. Scott Fitzgerald"),
        ("The Stranger", "Albert Camus"),
        ("One Hundred Years of Solitude", "Gabriel Garcia Marquez"),
        ("The Sun Also Rises", "Ernest Hemingway"),
        ("Middlemarch", "George Eliot"),
        ("The Color Purple", "Alice Walker"),
        ("To the Lighthouse", "Virginia Woolf"),
        ("A Tale of Two Cities", "Charles Dickens"),
        ("The Handmaid's Tale", "Margaret Atwood"),
        ("Things Fall Apart", "Chinua Achebe"),
        ("The Brothers Karamazov", "Fyodor Dostoevsky"),
    )
    return tuple({"work": work, "author": author} for work, author in facts)


def _element_records() -> tuple[dict[str, Any], ...]:
    facts = (
        ("hydrogen", "H"),
        ("oxygen", "O"),
        ("carbon", "C"),
        ("sodium", "Na"),
        ("iron", "Fe"),
        ("silver", "Ag"),
        ("gold", "Au"),
        ("potassium", "K"),
        ("nitrogen", "N"),
        ("chlorine", "Cl"),
        ("calcium", "Ca"),
        ("magnesium", "Mg"),
        ("copper", "Cu"),
        ("zinc", "Zn"),
        ("silicon", "Si"),
        ("phosphorus", "P"),
        ("helium", "He"),
        ("neon", "Ne"),
        ("tin", "Sn"),
        ("lead", "Pb"),
    )
    return tuple({"element": element, "symbol": symbol} for element, symbol in facts)


def _moon_records() -> tuple[dict[str, Any], ...]:
    facts = (
        ("Jupiter", "Ganymede"),
        ("Saturn", "Titan"),
        ("Earth", "the Moon"),
        ("Mars", "Phobos"),
        ("Neptune", "Triton"),
        ("Uranus", "Titania"),
        ("Pluto", "Charon"),
        ("Jupiter", "Europa"),
        ("Saturn", "Enceladus"),
        ("Mars", "Deimos"),
        ("Uranus", "Oberon"),
        ("Neptune", "Proteus"),
        ("Jupiter", "Io"),
        ("Saturn", "Rhea"),
        ("Uranus", "Miranda"),
        ("Neptune", "Nereid"),
        ("Saturn", "Dione"),
        ("Jupiter", "Callisto"),
        ("Uranus", "Ariel"),
        ("Saturn", "Iapetus"),
    )
    return tuple({"planet": planet, "moon": moon} for planet, moon in facts)


def _arithmetic_story_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("Mira", "she", "pencils", 7, 2, 3),
        ("Jonah", "he", "marbles", 12, 5, 4),
        ("Avery", "they", "tickets", 18, 6, 2),
        ("Leah", "she", "stickers", 11, 3, 5),
        ("Noah", "he", "notebooks", 9, 4, 7),
        ("Priya", "she", "berries", 14, 2, 6),
        ("Owen", "he", "blocks", 20, 8, 1),
        ("Clara", "she", "beads", 16, 7, 4),
        ("Elias", "he", "coins", 13, 5, 9),
        ("Mina", "she", "shells", 10, 1, 8),
        ("Lucas", "he", "cards", 15, 6, 6),
        ("Sara", "she", "markers", 19, 9, 2),
        ("Tariq", "he", "lemons", 8, 3, 10),
        ("Nina", "she", "erasers", 17, 4, 5),
        ("Caleb", "he", "cups", 21, 7, 3),
        ("Ivy", "she", "ribbons", 12, 2, 11),
        ("Mason", "he", "postcards", 18, 8, 4),
        ("Rhea", "she", "candles", 9, 2, 7),
        ("Dylan", "he", "spoons", 22, 10, 5),
        ("Aria", "she", "buttons", 14, 6, 9),
    )
    return tuple(
        {
            "name": name,
            "pronoun": pronoun,
            "item": item,
            "start": start,
            "give": give,
            "buy": buy,
        }
        for name, pronoun, item, start, give, buy in raw
    )


def _schedule_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("the shuttle", "7:10 AM", 35),
        ("the train", "3:25 PM", 45),
        ("the ferry", "11:40 AM", 50),
        ("the lecture", "1:15 PM", 90),
        ("the movie", "6:05 PM", 125),
        ("the workshop", "9:30 AM", 75),
        ("the rehearsal", "4:20 PM", 55),
        ("the meeting", "10:45 AM", 40),
        ("the bus", "8:05 AM", 65),
        ("the clinic", "2:30 PM", 30),
        ("the concert", "7:45 PM", 110),
        ("the practice", "5:10 PM", 80),
        ("the seminar", "12:20 PM", 70),
        ("the flight", "6:50 AM", 95),
        ("the webinar", "3:05 PM", 60),
        ("the debate", "1:40 PM", 85),
        ("the parade", "9:15 AM", 120),
        ("the shift", "11:05 PM", 45),
        ("the ceremony", "4:55 PM", 50),
        ("the pickup", "2:10 PM", 35),
    )
    return tuple(
        {"event": event, "start_time": start_time, "duration_minutes": duration}
        for event, start_time, duration in raw
    )


def _sequence_records() -> tuple[dict[str, Any], ...]:
    sequences = (
        "2, 4, 6, 8",
        "3, 6, 9, 12",
        "5, 10, 15, 20",
        "1, 4, 9, 16",
        "8, 16, 24, 32",
        "7, 14, 21, 28",
        "10, 20, 30, 40",
        "1, 3, 6, 10",
        "9, 18, 27, 36",
        "4, 8, 12, 16",
        "6, 12, 18, 24",
        "11, 22, 33, 44",
        "2, 5, 8, 11",
        "13, 26, 39, 52",
        "1, 2, 4, 8",
        "14, 21, 28, 35",
        "12, 24, 36, 48",
        "3, 7, 11, 15",
        "16, 25, 36, 49",
        "20, 40, 60, 80",
    )
    return tuple({"sequence": sequence} for sequence in sequences)


def _comparison_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("bottle", 750, "pitcher", 2000),
        ("backpack", 6, "duffel bag", 18),
        ("flash drive", 32, "external drive", 512),
        ("ladder", 8, "scaffold", 12),
        ("notebook", 120, "binder", 240),
        ("jar", 500, "bucket", 5000),
        ("bike trip", 14, "train ride", 55),
        ("hallway", 12, "gym", 28),
        ("garden bed", 9, "orchard row", 30),
        ("desk lamp", 40, "floor lamp", 90),
        ("mug", 350, "thermos", 950),
        ("carton", 6, "crate", 24),
        ("tablet", 128, "laptop", 1024),
        ("bridge", 60, "tunnel", 180),
        ("pond", 4, "reservoir", 25),
        ("chapel", 18, "auditorium", 400),
        ("booklet", 24, "manual", 180),
        ("cabin", 45, "lodge", 220),
        ("printer queue", 9, "batch export", 37),
        ("thermometer reading", 18, "greenhouse reading", 31),
    )
    return tuple(
        {
            "item_a": item_a,
            "amount_a": amount_a,
            "item_b": item_b,
            "amount_b": amount_b,
        }
        for item_a, amount_a, item_b, amount_b in raw
    )


def _python_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("clean_names", "names", "name", "strip"),
        ("lowercase_tags", "tags", "tag", "lower"),
        ("titlecase_titles", "titles", "title", "title"),
        ("normalize_codes", "codes", "code", "strip"),
        ("uppercase_states", "states", "state", "upper"),
        ("trim_labels", "labels", "label", "strip"),
        ("slugify_topics", "topics", "topic", "lower"),
        ("clean_paths", "paths", "path", "strip"),
        ("uppercase_grades", "grades", "grade", "upper"),
        ("normalize_words", "words", "word", "lower"),
        ("trim_fields", "fields", "field", "strip"),
        ("titlecase_cities", "cities", "city", "title"),
        ("clean_headers", "headers", "header", "strip"),
        ("uppercase_codes", "codes", "item", "upper"),
        ("normalize_names", "entries", "entry", "strip"),
        ("lowercase_keys", "keys", "key", "lower"),
        ("titlecase_labels", "labels", "label", "title"),
        ("clean_columns", "columns", "column", "strip"),
        ("uppercase_roles", "roles", "role", "upper"),
        ("normalize_notes", "notes", "note", "strip"),
    )
    return tuple(
        {
            "function_name": function_name,
            "collection": collection,
            "item_var": item_var,
            "method": method,
        }
        for function_name, collection, item_var, method in raw
    )


def _sql_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("orders", "customer_id", "total_amount", "created_at"),
        ("tickets", "agent_id", "resolution_minutes", "closed_at"),
        ("shipments", "region", "delivery_days", "shipped_at"),
        ("sessions", "user_id", "duration_seconds", "started_at"),
        ("invoices", "account_id", "amount_due", "issued_at"),
        ("products", "category", "unit_price", "updated_at"),
        ("scores", "team_id", "points", "recorded_at"),
        ("calls", "representative", "hold_seconds", "started_at"),
        ("messages", "channel_id", "message_count", "sent_at"),
        ("experiments", "variant", "conversion_rate", "created_at"),
        ("sales", "store_id", "revenue", "sold_at"),
        ("returns", "reason_code", "refund_amount", "processed_at"),
        ("users", "country", "login_count", "last_seen_at"),
        ("expenses", "department", "cost_usd", "approved_at"),
        ("weather_readings", "station_id", "temperature_c", "observed_at"),
        ("alerts", "severity", "response_minutes", "opened_at"),
        ("ship_logs", "captain_id", "delay_hours", "logged_at"),
        ("visits", "page_id", "visit_count", "visited_at"),
        ("payments", "method", "payment_amount", "captured_at"),
        ("forecasts", "region", "rainfall_mm", "generated_at"),
    )
    return tuple(
        {
            "table": table,
            "group_column": group_column,
            "measure_column": measure_column,
            "order_column": order_column,
        }
        for table, group_column, measure_column, order_column in raw
    )


def _shell_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("logs", "*.log", "logs-backup", "nginx"),
        ("reports", "*.csv", "reports-archive", "postgresql"),
        ("images", "*.png", "images-pack", "docker"),
        ("configs", "*.yaml", "configs-snapshot", "redis"),
        ("scripts", "*.sh", "scripts-bundle", "ssh"),
        ("snapshots", "*.json", "snapshots-pack", "cron"),
        ("exports", "*.tsv", "exports-rollup", "celery"),
        ("backups", "*.sql", "backups-weekly", "mysql"),
        ("docs", "*.md", "docs-sync", "airflow"),
        ("notebooks", "*.ipynb", "notebooks-pack", "jupyter"),
        ("assets", "*.svg", "assets-artifact", "grafana"),
        ("metrics", "*.prom", "metrics-dump", "prometheus"),
        ("captures", "*.pcap", "captures-pack", "suricata"),
        ("secrets", "*.env", "secrets-snapshot", "vault"),
        ("fixtures", "*.json", "fixtures-pack", "pytest"),
        ("audio", "*.wav", "audio-rollup", "ffmpeg"),
        ("videos", "*.mp4", "videos-archive", "plex"),
        ("queues", "*.txt", "queues-pack", "rabbitmq"),
        ("jobs", "*.toml", "jobs-snapshot", "systemd"),
        ("cache", "*.bin", "cache-rollup", "memcached"),
    )
    return tuple(
        {
            "directory": directory,
            "pattern": pattern,
            "archive_name": archive_name,
            "service": service,
        }
        for directory, pattern, archive_name, service in raw
    )


def _procedure_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("air filter", "unit", "panel"),
        ("kitchen scale", "reference weight", "display"),
        ("report export", "summary page", "file name"),
        ("bike chain", "rear wheel", "master link"),
        ("printer toner", "front cover", "release tab"),
        ("photo backup", "settings panel", "cloud sync"),
        ("router reset", "power cable", "status light"),
        ("lab balance", "sample tray", "tare button"),
        ("invoice approval", "totals section", "approver name"),
        ("plant repotting", "new pot", "drainage layer"),
        ("shipping label", "address block", "postal code"),
        ("calendar invite", "time zone", "location field"),
        ("project archive", "permissions page", "retention setting"),
        ("coffee grinder", "hopper lid", "grind selector"),
        ("camera battery", "charging dock", "indicator light"),
        ("travel claim", "receipt scan", "expense category"),
        ("whiteboard setup", "marker tray", "eraser"),
        ("audio mixer", "channel strip", "gain knob"),
        ("safety checklist", "inspection form", "signature line"),
        ("garden hose", "shutoff valve", "spray nozzle"),
    )
    return tuple(
        {"process": process, "object": object_name, "location": location}
        for process, object_name, location in raw
    )


def _scene_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("street vendor", "a crate of lemons", "market gate"),
        ("gardener", "a watering can", "brick path"),
        ("librarian", "a book cart", "reading room"),
        ("cyclist", "a helmet", "cafe window"),
        ("painter", "a ladder", "courtyard wall"),
        ("musician", "a stack of sheet music", "subway platform"),
        ("caretaker", "a key ring", "museum hall"),
        ("barista", "a stack of cups", "espresso machine"),
        ("baker", "a tray of rolls", "cooling rack"),
        ("mechanic", "a wrench", "garage bench"),
        ("teacher", "an attendance list", "classroom door"),
        ("florist", "a bucket of tulips", "shop counter"),
        ("runner", "a water bottle", "park bench"),
        ("photographer", "a camera strap", "harbor railing"),
        ("tailor", "a spool of thread", "cutting table"),
        ("chef", "a pan of herbs", "prep station"),
        ("custodian", "a mop bucket", "tile corridor"),
        ("vendor", "a string of paper lanterns", "festival arch"),
        ("student", "a laptop bag", "campus steps"),
        ("carpenter", "a wooden plank", "work shed"),
    )
    return tuple(
        {
            "actor": actor,
            "actor_article": _indefinite_article(actor),
            "actor_article_cap": _indefinite_article(actor).capitalize(),
            "object_phrase": object_phrase,
            "location": location,
        }
        for actor, object_phrase, location in raw
    )


def _workplace_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("nurse", "a clipboard", "ward station"),
        ("engineer", "a blueprint", "drafting table"),
        ("archivist", "a box of files", "storage room"),
        ("editor", "a style guide", "news desk"),
        ("designer", "a fabric swatch", "design studio"),
        ("chemist", "a glass beaker", "chemistry lab"),
        ("coach", "a stopwatch", "sideline table"),
        ("dentist", "a tray of tools", "exam room"),
        ("cashier", "a receipt roll", "checkout counter"),
        ("welder", "a protective visor", "welding booth"),
        ("analyst", "a spreadsheet printout", "conference table"),
        ("pilot", "a checklist", "cockpit"),
        ("chef", "an order ticket", "service rail"),
        ("botanist", "a sample tray", "greenhouse"),
        ("judge", "a case file", "bench chamber"),
        ("translator", "a glossary notebook", "desk lamp"),
        ("technician", "a tool case", "server rack"),
        ("curator", "a display label", "gallery cabinet"),
        ("dispatcher", "a radio handset", "control desk"),
        ("optician", "a lens tray", "optical shop"),
    )
    return tuple(
        {
            "profession": profession,
            "profession_article": _indefinite_article(profession),
            "profession_article_cap": _indefinite_article(profession).capitalize(),
            "tool_phrase": tool_phrase,
            "workspace": workspace,
        }
        for profession, tool_phrase, workspace in raw
    )


def _nature_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("fog", "riverbank", "silver"),
        ("sunlight", "pine ridge", "gold"),
        ("mist", "meadow grass", "blue"),
        ("rain", "stone walkway", "dark"),
        ("snow", "cedar branches", "white"),
        ("wind", "lake surface", "rippled"),
        ("dew", "orchard rows", "bright"),
        ("cloud shadow", "valley floor", "cool"),
        ("moonlight", "harbor water", "pale"),
        ("thunder", "distant hills", "restless"),
        ("dust", "desert road", "red"),
        ("spray", "waterfall rocks", "slick"),
        ("heat", "asphalt lot", "shimmering"),
        ("ice", "marsh reeds", "glassy"),
        ("starlight", "mountain pass", "clear"),
        ("hail", "barn roof", "loud"),
        ("breeze", "vine leaves", "soft"),
        ("sunset", "wheat field", "copper"),
        ("drizzle", "bus shelter", "dim"),
        ("frost", "garden fence", "quiet"),
    )
    return tuple(
        {"weather": weather, "location": location, "adjective": adjective}
        for weather, location, adjective in raw
    )


def _community_records() -> tuple[dict[str, Any], ...]:
    raw = (
        ("neighbors", "folding chairs", "sidewalk"),
        ("children", "chalk drawings", "playground wall"),
        ("volunteers", "cardboard boxes", "food pantry"),
        ("parents", "soccer bags", "parking lot"),
        ("guests", "paper programs", "entry table"),
        ("residents", "potted herbs", "balcony rail"),
        ("friends", "board game pieces", "living room rug"),
        ("visitors", "museum maps", "front desk"),
        ("students", "poster boards", "commons area"),
        ("workers", "orange cones", "service lane"),
        ("campers", "sleeping bags", "tent floor"),
        ("neighbors", "string lights", "fence line"),
        ("guests", "teacups", "dining room shelf"),
        ("children", "winter boots", "mudroom bench"),
        ("residents", "laundry baskets", "hall closet"),
        ("families", "picnic blankets", "grassy hill"),
        ("friends", "wrapped gifts", "coffee table"),
        ("organizers", "name tags", "check-in stand"),
        ("commuters", "umbrellas", "station bench"),
        ("guests", "coats", "wooden rack"),
    )
    return tuple(
        {"people": people, "object": object_name, "place": place}
        for people, object_name, place in raw
    )


def _factual_recall_subcategories() -> tuple[SubcategorySpec, ...]:
    return (
        SubcategorySpec(
            name="capital_fact",
            records=_capital_records(),
            template_pairs=(
                TemplatePair(
                    text="The capital city of {country} is",
                    paraphrase="{country}'s capital city is",
                ),
                TemplatePair(
                    text="Travel guides note that the capital of {country} is",
                    paraphrase="A travel guide would say that the capital of {country} is",
                ),
                TemplatePair(
                    text="On most maps, the capital of {country} appears as",
                    paraphrase="Most maps mark the capital of {country} as",
                ),
                TemplatePair(
                    text="In a geography quiz, the capital of {country} would be",
                    paraphrase="A geography quiz answer for the capital of {country} would be",
                ),
            ),
        ),
        SubcategorySpec(
            name="author_fact",
            records=_author_records(),
            template_pairs=(
                TemplatePair(
                    text="The novel {work} was written by",
                    paraphrase="{work} was written by",
                ),
                TemplatePair(
                    text="Literature students learn that {work} was written by",
                    paraphrase="Students in literature class learn that {work} was written by",
                ),
                TemplatePair(
                    text="The author of {work} is",
                    paraphrase="{work} was authored by",
                ),
                TemplatePair(
                    text="Most library catalogs list {work} under",
                    paraphrase="Library catalogs usually file {work} under",
                ),
            ),
        ),
        SubcategorySpec(
            name="element_symbol",
            records=_element_records(),
            template_pairs=(
                TemplatePair(
                    text="The chemical symbol for {element} is",
                    paraphrase="{element} has the chemical symbol",
                ),
                TemplatePair(
                    text="In the periodic table, {element} is abbreviated as",
                    paraphrase="The periodic-table abbreviation for {element} is",
                ),
                TemplatePair(
                    text="Chemistry notes write the symbol for {element} as",
                    paraphrase="A chemistry notebook would write the symbol for {element} as",
                ),
                TemplatePair(
                    text="A lab chart would mark {element} with the symbol",
                    paraphrase="On a lab chart, {element} would be marked with the symbol",
                ),
            ),
        ),
        SubcategorySpec(
            name="moon_fact",
            records=_moon_records(),
            template_pairs=(
                TemplatePair(
                    text="The best-known major moon orbiting {planet} is",
                    paraphrase="{planet}'s best-known major moon is",
                ),
                TemplatePair(
                    text="Astronomy students learn that a major moon of {planet} is",
                    paraphrase="Students in astronomy often learn that a major moon of {planet} is",
                ),
                TemplatePair(
                    text="A solar-system chart would label a major moon of {planet} as",
                    paraphrase="On a solar-system chart, a major moon of {planet} would be labeled",
                ),
                TemplatePair(
                    text="One widely recognized moon associated with {planet} is",
                    paraphrase="A widely recognized moon linked to {planet} is",
                ),
            ),
        ),
    )


def _reasoning_math_subcategories() -> tuple[SubcategorySpec, ...]:
    return (
        SubcategorySpec(
            name="arithmetic_story",
            records=_arithmetic_story_records(),
            template_pairs=(
                TemplatePair(
                    text="{name} had {start} {item}, gave away {give}, and bought {buy} more. {name} now has",
                    paraphrase="{name} started with {start} {item}, gave away {give}, and later bought {buy} more. {name} now has",
                ),
                TemplatePair(
                    text="After giving away {give} of the {start} {item} and then buying {buy} more, {name} has",
                    paraphrase="{name} began with {start} {item}. After giving away {give} and buying {buy} more, {name} has",
                ),
                TemplatePair(
                    text="{name} counted {start} {item}, lost {give}, and added {buy} later. The new total is",
                    paraphrase="The new total is what {name} has after starting with {start} {item}, losing {give}, and adding {buy}",
                ),
                TemplatePair(
                    text="{name} starts with {start} {item}. If {name} gives away {give} and gets {buy} back later, the total becomes",
                    paraphrase="{name} starts with {start} {item}; after giving away {give} and getting {buy} later, the total becomes",
                ),
            ),
        ),
        SubcategorySpec(
            name="schedule_reasoning",
            records=_schedule_records(),
            template_pairs=(
                TemplatePair(
                    text="If {event} begins at {start_time} and lasts {duration_minutes} minutes, it ends at",
                    paraphrase="{event} starts at {start_time} and lasts {duration_minutes} minutes, so it ends at",
                ),
                TemplatePair(
                    text="{event} starts at {start_time}. {duration_minutes} minutes later, the time will be",
                    paraphrase="{event} begins at {start_time}. After {duration_minutes} minutes, the time will be",
                ),
                TemplatePair(
                    text="A schedule shows {event} at {start_time} for {duration_minutes} minutes. The finish time is",
                    paraphrase="The schedule lists {event} at {start_time} for {duration_minutes} minutes, so the finish time is",
                ),
                TemplatePair(
                    text="When {event} begins at {start_time} and runs for {duration_minutes} minutes, it wraps up at",
                    paraphrase="{event} begins at {start_time} and runs for {duration_minutes} minutes, which means it wraps up at",
                ),
            ),
        ),
        SubcategorySpec(
            name="number_sequence",
            records=_sequence_records(),
            template_pairs=(
                TemplatePair(
                    text="The sequence {sequence} continues with",
                    paraphrase="If the sequence {sequence} keeps going, the next value is",
                ),
                TemplatePair(
                    text="A worksheet lists {sequence}; the next number would be",
                    paraphrase="On a worksheet, the pattern {sequence} would continue with",
                ),
                TemplatePair(
                    text="Continuing the pattern {sequence}, the next term is",
                    paraphrase="The next term after {sequence} is",
                ),
                TemplatePair(
                    text="If the pattern {sequence} is extended one more step, it becomes",
                    paraphrase="Extend the pattern {sequence} by one step and it becomes",
                ),
            ),
        ),
        SubcategorySpec(
            name="magnitude_comparison",
            records=_comparison_records(),
            template_pairs=(
                TemplatePair(
                    text="A {item_a} holds {amount_a} units and a {item_b} holds {amount_b} units. The larger one is the",
                    paraphrase="The {item_a} holds {amount_a} units and the {item_b} holds {amount_b} units. The larger one is the",
                ),
                TemplatePair(
                    text="Between the {item_a} at {amount_a} units and the {item_b} at {amount_b} units, the bigger option is the",
                    paraphrase="Compare a {item_a} with {amount_a} units to a {item_b} with {amount_b} units. The bigger option is the",
                ),
                TemplatePair(
                    text="A note lists the {item_a} as {amount_a} units and the {item_b} as {amount_b} units. The larger item is the",
                    paraphrase="The note says the {item_a} is {amount_a} units and the {item_b} is {amount_b} units, so the larger item is the",
                ),
                TemplatePair(
                    text="If one {item_a} measures {amount_a} units while one {item_b} measures {amount_b} units, the larger object is the",
                    paraphrase="One {item_a} measures {amount_a} units and one {item_b} measures {amount_b} units, making the larger object the",
                ),
            ),
        ),
    )


def _code_procedural_subcategories() -> tuple[SubcategorySpec, ...]:
    return (
        SubcategorySpec(
            name="python_snippet",
            records=_python_records(),
            template_pairs=(
                TemplatePair(
                    text="def {function_name}({collection}):\n    return [{item_var}.{method}() for {item_var} in",
                    paraphrase="def {function_name}({collection}):\n    return [value.{method}() for value in",
                ),
                TemplatePair(
                    text="def {function_name}({collection}):\n    cleaned = []\n    for {item_var} in {collection}:\n        cleaned.append({item_var}.{method}())\n    return",
                    paraphrase="def {function_name}({collection}):\n    result = []\n    for value in {collection}:\n        result.append(value.{method}())\n    return",
                ),
                TemplatePair(
                    text="def {function_name}({collection}):\n    return list(map(lambda {item_var}: {item_var}.{method}(),",
                    paraphrase="def {function_name}({collection}):\n    return list(map(lambda value: value.{method}(),",
                ),
                TemplatePair(
                    text="{function_name} = lambda {collection}: [{item_var}.{method}() for {item_var} in",
                    paraphrase="{function_name} = lambda {collection}: [value.{method}() for value in",
                ),
            ),
        ),
        SubcategorySpec(
            name="sql_snippet",
            records=_sql_records(),
            template_pairs=(
                TemplatePair(
                    text="SELECT {group_column}, COUNT(*)\nFROM {table}\nGROUP BY",
                    paraphrase="SELECT COUNT(*), {group_column}\nFROM {table}\nGROUP BY",
                ),
                TemplatePair(
                    text="SELECT {group_column}, AVG({measure_column})\nFROM {table}\nGROUP BY",
                    paraphrase="SELECT AVG({measure_column}), {group_column}\nFROM {table}\nGROUP BY",
                ),
                TemplatePair(
                    text="SELECT {group_column}, MAX({measure_column})\nFROM {table}\nGROUP BY",
                    paraphrase="SELECT MAX({measure_column}), {group_column}\nFROM {table}\nGROUP BY",
                ),
                TemplatePair(
                    text="SELECT {group_column}\nFROM {table}\nORDER BY {order_column} DESC,",
                    paraphrase="SELECT {group_column}\nFROM {table}\nORDER BY {order_column} DESC NULLS LAST,",
                ),
            ),
        ),
        SubcategorySpec(
            name="shell_snippet",
            records=_shell_records(),
            template_pairs=(
                TemplatePair(
                    text="find {directory} -name '{pattern}' -print |",
                    paraphrase="find {directory} -type f -name '{pattern}' -print |",
                ),
                TemplatePair(
                    text="grep -R '{pattern}' {directory} |",
                    paraphrase="grep -R --line-number '{pattern}' {directory} |",
                ),
                TemplatePair(
                    text="tar -czf {archive_name}.tar.gz {directory}/",
                    paraphrase="tar -czf {archive_name}.tar.gz ./{directory}/",
                ),
                TemplatePair(
                    text="systemctl status {service} |",
                    paraphrase="systemctl --no-pager status {service} |",
                ),
            ),
        ),
        SubcategorySpec(
            name="procedure_text",
            records=_procedure_records(),
            template_pairs=(
                TemplatePair(
                    text="To service the {process}, first place the {object} near the {location} and then check the",
                    paraphrase="When servicing the {process}, first place the {object} near the {location} and then check the",
                ),
                TemplatePair(
                    text="Before finishing the {process}, set the {object} beside the {location} and review the",
                    paraphrase="Set the {object} beside the {location} before finishing the {process}, then review the",
                ),
                TemplatePair(
                    text="A technician completing the {process} would position the {object} near the {location} before opening the",
                    paraphrase="To complete the {process}, a technician would place the {object} near the {location} before opening the",
                ),
                TemplatePair(
                    text="The checklist for the {process} says to keep the {object} by the {location} and then inspect the",
                    paraphrase="According to the checklist for the {process}, keep the {object} by the {location} and then inspect the",
                ),
            ),
        ),
    )


def _general_text_subcategories() -> tuple[SubcategorySpec, ...]:
    return (
        SubcategorySpec(
            name="street_scene",
            records=_scene_records(),
            template_pairs=(
                TemplatePair(
                    text="Near the {location}, {actor_article} {actor} set {object_phrase} beside the",
                    paraphrase="{actor_article_cap} {actor} near the {location} placed {object_phrase} beside the",
                ),
                TemplatePair(
                    text="By the {location}, the {actor} balanced {object_phrase} on the",
                    paraphrase="The {actor} balanced {object_phrase} on the edge of the {location} beside the",
                ),
                TemplatePair(
                    text="At the {location}, {actor_article} {actor} reached for {object_phrase} under the",
                    paraphrase="{actor_article_cap} {actor} at the {location} reached under the awning for {object_phrase} near the",
                ),
                TemplatePair(
                    text="{actor_article_cap} {actor} passed the {location} carrying {object_phrase} toward the",
                    paraphrase="Carrying {object_phrase}, {actor_article} {actor} passed the {location} toward the",
                ),
            ),
        ),
        SubcategorySpec(
            name="workplace_scene",
            records=_workplace_records(),
            template_pairs=(
                TemplatePair(
                    text="At the {workspace}, {profession_article} {profession} set {tool_phrase} beside the",
                    paraphrase="{profession_article_cap} {profession} set {tool_phrase} beside the bench at the {workspace}",
                ),
                TemplatePair(
                    text="At the {workspace}, {profession_article} {profession} reached for {tool_phrase} near the",
                    paraphrase="{profession_article_cap} {profession} at the {workspace} reached near the cabinet for {tool_phrase}",
                ),
                TemplatePair(
                    text="The {profession} checked {tool_phrase} before crossing the {workspace} toward the",
                    paraphrase="Before crossing the {workspace}, the {profession} checked {tool_phrase} near the",
                ),
                TemplatePair(
                    text="At the {workspace}, {profession_article} {profession} rested {tool_phrase} on the",
                    paraphrase="{profession_article_cap} {profession} at the {workspace} rested {tool_phrase} on the counter by the",
                ),
            ),
        ),
        SubcategorySpec(
            name="nature_scene",
            records=_nature_records(),
            template_pairs=(
                TemplatePair(
                    text="After the {weather} passed over the {location}, everything looked",
                    paraphrase="Everything looked {adjective} after the {weather} moved over the {location}",
                ),
                TemplatePair(
                    text="Across the {location}, the {weather} made the air feel",
                    paraphrase="The air over the {location} felt {adjective} once the {weather} moved through",
                ),
                TemplatePair(
                    text="By the {location}, the {weather} left the landscape looking",
                    paraphrase="The landscape by the {location} looked {adjective} after the {weather}",
                ),
                TemplatePair(
                    text="The {weather} above the {location} turned the whole scene",
                    paraphrase="Above the {location}, the {weather} turned the whole scene {adjective}",
                ),
            ),
        ),
        SubcategorySpec(
            name="community_scene",
            records=_community_records(),
            template_pairs=(
                TemplatePair(
                    text="At the {place}, the {people} stacked {object} near the",
                    paraphrase="The {people} stacked {object} near the entrance of the {place}",
                ),
                TemplatePair(
                    text="Inside the {place}, the {people} arranged {object} beside the",
                    paraphrase="The {people} arranged {object} beside the table inside the {place}",
                ),
                TemplatePair(
                    text="By the {place}, the {people} carried {object} toward the",
                    paraphrase="Carrying {object}, the {people} moved through the {place} toward the",
                ),
                TemplatePair(
                    text="The {people} left {object} resting near the {place}'s",
                    paraphrase="Near the {place}, the {people} left {object} resting by the",
                ),
            ),
        ),
    )


def _build_entries_for_stratum(
    *,
    stratum_name: str,
    subcategories: tuple[SubcategorySpec, ...],
) -> dict[str, list[dict[str, Any]]]:
    prompts_by_split: dict[str, list[dict[str, Any]]] = {"pilot": [], "confirm": []}
    counters = {"pilot": 0, "confirm": 0}
    tags_prefix = ["oracle_alpha", f"stratum_{stratum_name}"]

    for subcategory in subcategories:
        pilot_records, confirm_records = _split_records(subcategory.records)
        for split, records in (("pilot", pilot_records), ("confirm", confirm_records)):
            for record in records:
                for template_pair in subcategory.template_pairs:
                    counters[split] += 1
                    prompt_id = f"oa5-{split}-{stratum_name}-{counters[split]:03d}"
                    prompt = {
                        "id": prompt_id,
                        "split": split,
                        "text": template_pair.text.format(**record),
                        "tags": tags_prefix + [f"subcategory_{subcategory.name}"],
                    }
                    if split == "pilot":
                        prompt["perturbations"] = {
                            "prompt_paraphrase": template_pair.paraphrase.format(
                                **record
                            )
                        }
                    prompts_by_split[split].append(prompt)

    expected_counts = {
        "pilot": len(subcategories)
        * PILOT_RECORDS_PER_SUBCATEGORY
        * len(subcategories[0].template_pairs),
        "confirm": len(subcategories)
        * CONFIRM_RECORDS_PER_SUBCATEGORY
        * len(subcategories[0].template_pairs),
    }
    for split, expected in expected_counts.items():
        actual = len(prompts_by_split[split])
        if actual != expected:
            raise ValueError(
                f"{stratum_name} {split} prompt count mismatch: {actual} != {expected}"
            )
    return prompts_by_split


def _round_robin(
    prompt_groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    ordered_keys = (
        "factual_recall",
        "reasoning_math",
        "code_procedural",
        "general_text",
    )
    max_length = max(len(prompt_groups[key]) for key in ordered_keys)
    ordered_prompts: list[dict[str, Any]] = []
    for index in range(max_length):
        for key in ordered_keys:
            group = prompt_groups[key]
            if index < len(group):
                ordered_prompts.append(group[index])
    return ordered_prompts


def build_registry_payload() -> dict[str, Any]:
    base_registry = yaml.safe_load(BASE_REGISTRY_PATH.read_text())
    collections = base_registry["collections"]
    oracle_collection = collections["oracle_alpha_phase1_v1"]

    stratum_builders = {
        "factual_recall": _factual_recall_subcategories(),
        "reasoning_math": _reasoning_math_subcategories(),
        "code_procedural": _code_procedural_subcategories(),
        "general_text": _general_text_subcategories(),
    }
    prompts_by_split_by_stratum = {
        stratum_name: _build_entries_for_stratum(
            stratum_name=stratum_name,
            subcategories=subcategories,
        )
        for stratum_name, subcategories in stratum_builders.items()
    }
    pilot_prompts = _round_robin(
        {
            stratum_name: split_prompts["pilot"]
            for stratum_name, split_prompts in prompts_by_split_by_stratum.items()
        }
    )
    confirm_prompts = _round_robin(
        {
            stratum_name: split_prompts["confirm"]
            for stratum_name, split_prompts in prompts_by_split_by_stratum.items()
        }
    )

    oracle_collection["description"] = (
        "Stratified primary-model oracle-alpha prompt surface that expands the "
        "saved oracle collection to four explicit strata with 256 pilot prompts "
        "and 1024 confirm prompts. The strata are factual recall, reasoning and "
        "math, code and procedural text, and general narrative or expository "
        "text. Pilot entries preserve saved paraphrases for stability checks, "
        "and split ordering is round-robin across strata so small calibration "
        "slices cover the full surface."
    )
    oracle_collection["prompts"] = pilot_prompts + confirm_prompts

    base_registry["version"] = 5
    base_registry["registry_id"] = "20260318-pilot-confirm-v5"
    base_registry["created_on"] = "2026-03-18"
    return base_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write the generated registry_v5 YAML.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_registry_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(payload, sort_keys=False, width=88))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
