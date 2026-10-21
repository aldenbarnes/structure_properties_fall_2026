"""Forgiving San Francisco address matching, checked against SF property records.

    idx = sf_address.AddressIndex.load()          # reads Data/properties.csv (~1 s)
    idx.match("1010 anza")                       # -> [Candidate(...1010 Anza St...), ...]

Typed input is cleaned up before matching: city, state and zip are optional, case and
punctuation don't matter, street types can be spelled out or abbreviated ("street",
"st", "avenue", "ave"), numbered streets work as "25th ave", "25 ave" or "twenty-fifth"
style ordinals up to tenth, and a unit ("#5", "apt 5", "unit 5") is split off.
Misspelled street names are matched to the closest real SF street name.

Standard library only.
"""
import csv
import difflib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "Data" / "properties.csv"
NEAR_NUMBERS = 50   # borrow lat/lon from a parcel this close in house number on the same street

# Assessor street-type codes, and what people type for each.
STREET_TYPES = {
    "ST": ["st", "str", "street"],
    "AV": ["av", "ave", "avenue", "avn"],
    "BL": ["bl", "blvd", "boulevard"],
    "DR": ["dr", "drive"],
    "WY": ["wy", "way"],
    "TE": ["te", "ter", "terr", "terrace"],
    "CT": ["ct", "court"],
    "PL": ["pl", "place"],
    "LN": ["ln", "lane"],
    "CR": ["cr", "cir", "circle"],
    "RD": ["rd", "road"],
    "AL": ["al", "aly", "alley"],
    "HY": ["hy", "hwy", "highway"],
    "PZ": ["pz", "plz", "plaza"],
    "PK": ["pk", "park"],
    "LP": ["lp", "loop"],
    "SQ": ["sq", "square"],
    "WK": ["wk", "walk"],
    "RW": ["rw", "row"],
    "HL": ["hl", "hill"],
    "TR": ["tr", "trl", "trail"],
}
TYPE_WORDS = {w: code for code, words in STREET_TYPES.items() for w in words}
PRETTY_TYPE = {"ST": "St", "AV": "Ave", "BL": "Blvd", "DR": "Dr", "WY": "Way", "TE": "Ter", "CT": "Ct",
               "PL": "Pl", "LN": "Ln", "CR": "Cir", "RD": "Rd", "AL": "Aly", "HY": "Hwy", "PZ": "Plz",
               "PK": "Park", "LP": "Loop", "SQ": "Sq", "WK": "Walk", "RW": "Row", "HL": "Hill", "TR": "Trl"}
ORDINAL_WORDS = {"first": "1ST", "second": "2ND", "third": "3RD", "fourth": "4TH", "fifth": "5TH",
                 "sixth": "6TH", "seventh": "7TH", "eighth": "8TH", "ninth": "9TH", "tenth": "10TH"}
DIRECTIONS = {"n": "NORTH", "s": "SOUTH", "e": "EAST", "w": "WEST"}
CITY_WORDS = r"\b(san\s*francisco|san\s*fran|sanfrancisco|sf|s\.f\.|california|calif|ca|usa|us)\b"
ASSESSOR_ADDR = re.compile(r"^(\d+)[A-Z]?\s+(.+?)\s(" + "|".join(STREET_TYPES) + r")(?=$|[\s\d])")


def _ordinal(n):
    n = int(n)
    suffix = "TH" if 10 <= n % 100 <= 20 else {1: "ST", 2: "ND", 3: "RD"}.get(n % 10, "TH")
    return "{}{}".format(n, suffix)


def _squash(s):
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def pretty_street(name, stype):
    words = []
    for w in name.split():
        if re.match(r"^\d+(ST|ND|RD|TH)$", w):
            words.append(w.lower())
        elif w.startswith("MC") and len(w) > 3:
            words.append("Mc" + w[2:].capitalize())      # MCALLISTER -> McAllister
        else:
            words.append(w.capitalize())
    return " ".join(words + ([PRETTY_TYPE.get(stype, stype.title())] if stype else []))


@dataclass
class Parsed:
    number: str = None       # house number, e.g. "1010"
    number_hi: str = None    # end of an address range, e.g. "1505" in "1501-1505 Masonic Ave"
    street: str = ""         # street name in assessor style, without type, e.g. "ANZA"
    stype: str = None        # assessor type code, e.g. "ST"; None if not typed
    unit: str = None
    zip_code: str = None


def parse(text):
    """Split free-typed SF address text into number, street name, street type, unit and zip."""
    s = " " + text.lower().strip() + " "
    p = Parsed()
    m = re.search(r"\b(9[0-9]{4})(?:-\d{4})?\b", s)
    if m:
        p.zip_code = m.group(1)
        s = s.replace(m.group(0), " ")
    m = re.search(r"(?:#|\bapt\b\.?|\bapartment\b|\bunit\b|\bste\b|\bsuite\b|\bno\b\.?)\s*([a-z0-9-]+)", s)
    if m:
        p.unit = m.group(1).upper()
        s = s[:m.start()] + " " + s[m.end():]
    s = re.sub(CITY_WORDS, " ", s)
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    toks = s.split()
    if toks and re.match(r"^\d+(-\d+)?[a-z]?$", toks[0]) and len(toks) > 1:
        m = re.match(r"^(\d+)(?:-(\d+))?", toks[0])
        p.number, p.number_hi = m.group(1), m.group(2)
        toks = toks[1:]
    # A trailing unit number like "1010 anza st 5" (only after a street type).
    if len(toks) >= 3 and toks[-2] in TYPE_WORDS and re.match(r"^[0-9]+[a-z]?$", toks[-1]) and not p.unit:
        p.unit = toks.pop().upper()
    if len(toks) >= 2 and toks[-1] in TYPE_WORDS:
        p.stype = TYPE_WORDS[toks.pop()]
    elif len(toks) >= 2 and len(toks[-1]) <= 3 and any(w.startswith(toks[-1]) for w in TYPE_WORDS):
        toks.pop()                                     # a street type still being typed: "1010 anza s"
    out = []
    for i, t in enumerate(toks):
        if t in ORDINAL_WORDS:
            out.append(ORDINAL_WORDS[t])
        elif re.match(r"^\d+$", t):
            out.append(_ordinal(t))                    # "25 ave" -> 25TH AV
        elif re.match(r"^\d+(st|nd|rd|th)$", t):
            out.append(t.upper())
        elif i == 0 and t in DIRECTIONS and len(toks) > 1:
            out.append(DIRECTIONS[t])                  # "n point st" -> NORTH POINT
        else:
            out.append(t.upper())
    p.street = " ".join(out)
    return p


@dataclass
class Building:
    number: str
    street: str
    stype: str
    latitude: float = None
    longitude: float = None
    neighborhood: str = None
    year_built: int = None
    units: int = None

    @property
    def address(self):
        street = pretty_street(self.street, self.stype)
        return "{} {}".format(self.number, street) if self.number else street

    @property
    def facts(self):
        """Short description from the property records, e.g. '13 units, built 1972, Inner Richmond'."""
        bits = []
        if self.units:
            bits.append("{} unit{}".format(self.units, "" if self.units == 1 else "s"))
        if self.year_built:
            bits.append("built {}".format(self.year_built))
        if self.neighborhood:
            bits.append(self.neighborhood)
        return ", ".join(bits)

    @property
    def query(self):
        """What to send to RealtyAPI's location search."""
        return self.address + ", San Francisco, CA"


@dataclass
class Candidate:
    building: Building
    score: float
    exact: bool                     # number + street found in the property records
    notes: list = field(default_factory=list)
    street_similarity: float = 1.0


def _num(x, cast=float):
    try:
        return cast(float(x))
    except (TypeError, ValueError):
        return None


class AddressIndex:
    def __init__(self, buildings):
        self.buildings = buildings                       # (number, street, stype) -> Building
        self.by_street = {}                              # (street, stype) -> {number: Building}
        for (num, street, stype), b in buildings.items():
            self.by_street.setdefault((street, stype), {})[num] = b
        self.street_names = sorted({s for s, _ in self.by_street})
        self.types_by_street = {}
        for s, t in self.by_street:
            self.types_by_street.setdefault(s, []).append(t)
        self._squashed = {}
        for s in self.street_names:
            self._squashed.setdefault(_squash(s), []).append(s)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        buildings = {}
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                m = ASSESSOR_ADDR.match((r.get("address_unit") or "").strip().upper())
                if not m:
                    continue
                key = (m.group(1), m.group(2), m.group(3))
                b = buildings.get(key)
                if b is None:
                    b = buildings[key] = Building(*key, latitude=_num(r.get("latitude")),
                                                  longitude=_num(r.get("longitude")),
                                                  neighborhood=r.get("analysis_neighborhood") or None,
                                                  year_built=_num(r.get("year_built"), int), units=0)
                b.units += _num(r.get("units"), int) or 0
        for b in buildings.values():
            b.units = b.units or None
        return cls(buildings)

    def _street_matches(self, street, limit=8):
        """Real street names closest to `street`, with a 0-1 similarity score."""
        if not street:
            return []
        target = _squash(street)
        if street in self.by_street_names_set():
            exact = [(street, 1.0)]
        else:
            exact = [(s, 0.99) for s in self._squashed.get(target, [])]   # "vanness" -> VAN NESS
        scored = {s: sc for s, sc in exact}
        for s in difflib.get_close_matches(street, self.street_names, n=limit, cutoff=0.6):
            scored.setdefault(s, difflib.SequenceMatcher(None, street, s).ratio())
        for sq in difflib.get_close_matches(target, list(self._squashed), n=limit, cutoff=0.7):
            for s in self._squashed[sq]:
                scored.setdefault(s, difflib.SequenceMatcher(None, target, sq).ratio() * 0.98)
        return sorted(scored.items(), key=lambda kv: -kv[1])[:limit]

    def by_street_names_set(self):
        if not hasattr(self, "_names_set"):
            self._names_set = set(self.street_names)
        return self._names_set

    def match(self, text, limit=6):
        """Best-matching real SF addresses for free-typed text, best first."""
        p = parse(text)
        out = []
        for street, sim in self._street_matches(p.street):
            for (s, stype), numbers in self.by_street.items():
                if s != street:
                    continue
                score = sim
                notes = []
                if p.stype and stype != p.stype:
                    score -= 0.08
                    notes.append("{} instead of {}".format(PRETTY_TYPE.get(stype, stype), PRETTY_TYPE.get(p.stype, p.stype)))
                if sim < 1.0:
                    notes.append("spelling corrected")
                # Street similarity dominates; the house number only breaks ties, because many
                # real addresses are filed under a neighboring number in the property records.
                if p.number is None:
                    b = Building(None, s, stype)
                    out.append(Candidate(b, score - 0.05, False, notes + ["no house number"], sim))
                elif p.number in numbers:
                    out.append(Candidate(numbers[p.number], score + 0.05, True, notes, sim))
                else:
                    near = min(numbers, key=lambda n: abs(int(n) - int(p.number)))
                    nb = numbers[near]
                    close = abs(int(near) - int(p.number)) <= NEAR_NUMBERS
                    b = Building(p.number, s, stype, nb.latitude if close else None,
                                 nb.longitude if close else None, nb.neighborhood if close else None)
                    out.append(Candidate(b, score - 0.05, False, notes + [
                        "{} is not in the property records (nearest: {})".format(p.number, near)], sim))
        out.sort(key=lambda c: -c.score)
        return out[:limit]

    def suggest(self, text, limit=8):
        """Address suggestions while typing: real SF addresses whose street starts with, or is close
        to, what was typed, best match first."""
        p = parse(text)
        if not p.street or p.number is None:
            return []
        target = _squash(p.street)
        streets = {s: (1.0 if _squash(s) == target else 0.97)
                   for s in self.street_names if _squash(s).startswith(target)}
        for s, sim in self._street_matches(p.street):
            streets.setdefault(s, sim)
        out = []
        for street, sim in streets.items():
            for stype in self.types_by_street[street]:
                numbers = self.by_street[(street, stype)]
                score = sim + (0.02 if p.stype == stype else -0.08 if p.stype else 0)
                if p.number in numbers:
                    out.append(Candidate(numbers[p.number], score + 0.05, True, [], sim))
                else:
                    near = min(numbers, key=lambda n: abs(int(n) - int(p.number)))
                    nb = numbers[near]
                    close = abs(int(near) - int(p.number)) <= NEAR_NUMBERS
                    b = Building(p.number, street, stype, nb.latitude if close else None,
                                 nb.longitude if close else None, nb.neighborhood if close else None)
                    out.append(Candidate(b, score - 0.05, False, ["not in property records"], sim))
        out.sort(key=lambda c: -c.score)
        return out[:limit]

    def is_confident(self, candidates):
        """True when the top candidate's street is a near-certain match and clearly the best one."""
        if not candidates or candidates[0].street_similarity < 0.85:
            return False
        return len(candidates) == 1 or candidates[1].score <= candidates[0].score - 0.08


def same_unit(typed, unit):
    """True if a typed unit ("5", "#05") is the listing's unit ("5", "Unit 5", "N/A-5")."""
    if not typed or not unit:
        return False
    norm = lambda u: re.sub(r"^0+", "", re.split(r"[\s#-]", u.strip().upper())[-1]) or "0"
    return norm(typed) == norm(unit)


def street_similarity(address_text, building):
    """0-1 similarity of a listing address's street name to the building's street."""
    p = parse(address_text or "")
    if not p.street:
        return 0.0
    return difflib.SequenceMatcher(None, _squash(p.street), _squash(building.street)).ratio()


def same_building(address_text, building):
    """0-1 score for whether a listing address (e.g. RealtyAPI oneLineAddress) is `building`.

    Full street similarity when the house number matches or falls inside an address range
    ("1501-1505 Masonic Ave" covers 1503); half when it is a few numbers away on the same side
    of the street (same odd/even parity), which alone is not enough to call it a match.
    """
    p = parse(address_text or "")
    if building.number is None or p.number is None:
        return 0.0
    street_sim = street_similarity(address_text, building)
    n, lo = int(building.number), int(p.number)
    hi = int(p.number_hi) if p.number_hi and int(p.number_hi) >= lo else lo
    if lo <= n <= hi:
        return street_sim
    if min(abs(n - lo), abs(n - hi)) <= 10 and n % 2 == lo % 2:
        return street_sim * 0.5
    return 0.0


def same_side(address_text, building):
    """False when a listing on the building's street is across the street (odd vs even numbers)."""
    p = parse(address_text or "")
    if building.number is None or p.number is None:
        return True
    return int(p.number) % 2 == int(building.number) % 2


class ZipGuesser:
    """Zip code for a location, from the nearest listings with known zips (training data and comps).

    SF property records have no zip. A zip helps apartments.com resolve an address, and is needed
    for the zip-wide search; a wrong guess near a zip boundary only makes that search less useful.
    """

    def __init__(self, points):
        self.points = points                     # [(lat, lon, zip)]

    @classmethod
    def load(cls, paths=None):
        data = Path(__file__).resolve().parent.parent / "Data"
        paths = paths or [data / "realty_training.csv", data / "RealtyAPI_1010_Anza_Comps.csv"]
        points = {}
        for path in paths:
            try:
                with open(path, newline="", encoding="utf-8-sig") as f:
                    for r in csv.DictReader(f):
                        lat, lon = _num(r.get("latitude")), _num(r.get("longitude"))
                        z = (r.get("zip_code") or "").strip()[:5]
                        if lat and lon and re.fullmatch(r"9\d{4}", z):
                            points[(round(lat, 5), round(lon, 5))] = z
            except OSError:
                continue
        return cls([(lat, lon, z) for (lat, lon), z in points.items()])

    def guess(self, lat, lon, max_m=600, k=5, agree=0.8):
        """The zip shared by at least `agree` of the k nearest known listings within max_m meters,
        or None (no listings nearby, or near a zip boundary where they disagree)."""
        if lat is None or lon is None or not self.points:
            return None
        coslat = math.cos(math.radians(lat))
        near = sorted((math.hypot((plat - lat) * 111_000, (plon - lon) * 111_000 * coslat), z)
                      for plat, plon, z in self.points)
        near = [z for d, z in near[:k] if d <= max_m]
        if not near:
            return None
        best = max(set(near), key=near.count)
        return best if near.count(best) >= agree * len(near) else None
