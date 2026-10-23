"""Strict, run-scoped adapter for direct API-Football v3 (no startup I/O)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import re

import requests

_BASE = "https://v3.football.api-sports.io"
_REASONS = {"authentication", "quota", "coverage", "ambiguous_league", "provider_error"}


class ApiFootballError(ValueError):
    """Safe public failure: provider payloads and credentials are never retained."""
    def __init__(self, reason="provider_error", status_code=None):
        self.reason = reason if reason in _REASONS else "provider_error"
        self.status_code = status_code if type(status_code) is int else None
        super().__init__(f"API-Football: {self.reason}")


def _positive(value):
    return type(value) is int and value > 0


def _timestamp(value):
    if not isinstance(value, (str, datetime)):
        raise ValueError("invalid time")
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError("UTC time required")
    return stamp.astimezone(timezone.utc)


class ApiFootballClient:
    def __init__(self, api_key, now=None):
        self._key = api_key
        self._now = now
        self._metadata = None
        self._fixtures = None
        self.quota = {}

    def _request(self, endpoint, params, page=1):
        if not isinstance(self._key, str) or not self._key.strip():
            raise ApiFootballError("authentication")
        if any(self.quota.get(key) == 0 for key in ("remaining", "minute_remaining")):
            raise ApiFootballError("quota")
        failed = False
        try:
            response = requests.get(_BASE + endpoint, params=params,
                                    headers={"x-apisports-key": self._key},
                                    timeout=30, allow_redirects=False)
        except requests.RequestException:
            failed = True
        if failed:
            raise ApiFootballError() from None
        status = response.status_code
        headers = {str(k).lower(): v for k, v in response.headers.items()}
        for key, header in (("remaining", "x-ratelimit-requests-remaining"),
                            ("limit", "x-ratelimit-requests-limit"),
                            ("minute_remaining", "x-ratelimit-remaining"),
                            ("minute_limit", "x-ratelimit-limit")):
            raw = headers.get(header)
            if isinstance(raw, str) and raw.isascii() and raw.isdigit():
                self.quota[key] = int(raw)
        if "limit" in self.quota and "remaining" in self.quota:
            self.quota["used"] = max(0, self.quota["limit"] - self.quota["remaining"])
        if not 200 <= status < 300:
            reason = "authentication" if status in (401, 403) else "quota" if status == 429 else "provider_error"
            raise ApiFootballError(reason, status)
        invalid = False
        try:
            payload = response.json()
        except (ValueError, requests.RequestException):
            invalid = True
        if invalid or not isinstance(payload, dict):
            raise ApiFootballError()
        errors = payload.get("errors")
        if errors:
            # Inspect only to classify; never include provider text in the exception.
            description = str(errors).lower()
            reason = ("authentication" if any(x in description for x in ("token", "key", "auth")) else
                      "quota" if any(x in description for x in ("limit", "quota", "requests")) else
                      "coverage" if any(x in description for x in ("plan", "subscription", "coverage", "access")) else
                      "provider_error")
            raise ApiFootballError(reason)
        paging, rows = payload.get("paging"), payload.get("response")
        if (not isinstance(errors, (dict, list)) or not isinstance(rows, list) or
                not isinstance(paging, dict) or type(paging.get("current")) is not int or
                paging["current"] != page or not _positive(paging.get("total")) or
                paging["total"] < page or paging["total"] > 20 or
                type(payload.get("results")) is not int or payload["results"] != len(rows)):
            raise ApiFootballError()
        return rows, paging["total"]

    def _pages(self, endpoint, params):
        result, total = self._request(endpoint, params)
        if total > 1 and not result:
            raise ApiFootballError()
        for page in range(2, total + 1):
            rows, actual_total = self._request(endpoint, dict(params, page=page), page)
            if actual_total != total or not rows:
                raise ApiFootballError()
            result.extend(rows)
        return result

    def _discover(self):
        if self._metadata is not None:
            return self._metadata
        rows = self._pages("/leagues", {"country": "Israel", "current": "true", "type": "League"})
        matches = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            info, country = row.get("league", {}), row.get("country", {})
            if not isinstance(info, dict) or not isinstance(country, dict):
                continue
            name = re.sub(r"[^a-z]", "", str(info.get("name", "")).lower())
            if country.get("name") == "Israel" and name == "ligathaal" and info.get("type") == "League":
                matches.append(row)
        if len(matches) != 1:
            raise ApiFootballError("ambiguous_league" if len(matches) > 1 else "coverage")
        row = matches[0]
        seasons = row.get("seasons")
        current = [s for s in seasons if isinstance(s, dict) and s.get("current") is True] if isinstance(seasons, list) else []
        if (not _positive(row["league"].get("id")) or len(current) != 1 or
                not _positive(current[0].get("year")) or not isinstance(current[0].get("coverage"), dict) or
                current[0]["coverage"].get("odds") is not True):
            raise ApiFootballError("coverage")
        self._metadata = row["league"]["id"], current[0]["year"]
        return self._metadata

    def _load_fixtures(self):
        if self._fixtures is not None:
            return self._fixtures
        league, season = self._discover()
        raw_now = self._now() if callable(self._now) else self._now
        try:
            now = _timestamp(raw_now) if raw_now is not None else datetime.now(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            raise ApiFootballError() from None
        rows = self._pages("/fixtures", {"league": league, "season": season,
                            "from": (now - timedelta(days=3)).date().isoformat(),
                            "to": (now + timedelta(days=14)).date().isoformat(), "timezone": "UTC"})
        fixtures = {}
        duplicates = set()
        for row in rows:
            try:
                info, teams = row["fixture"], row["teams"]
                home, away = teams["home"], teams["away"]
                identity = info["id"]
                status = info.get("status")
                if (not isinstance(status, dict) or not isinstance(status.get("short"), str)
                        or not status["short"].strip()):
                    continue
                if (not _positive(identity) or row["league"]["id"] != league or
                        type(row["league"]["id"]) is not int or row["league"]["season"] != season or
                        type(row["league"]["season"]) is not int or
                        not _positive(home["id"]) or not _positive(away["id"]) or home["id"] == away["id"] or
                        any(not isinstance(t["name"], str) or not t["name"].strip() for t in (home, away)) or
                        home["name"].strip() == away["name"].strip()):
                    continue
                kickoff = _timestamp(info["date"]).isoformat()
                if identity in fixtures:
                    duplicates.add(identity)
                fixtures[identity] = (row, {"id": f"api-football:{identity}", "commence_time": kickoff,
                                          "home_team": home["name"], "away_team": away["name"]})
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        self._fixtures = {k: v for k, v in fixtures.items() if k not in duplicates}
        return self._fixtures

    def fetch_scores(self):
        events = []
        for row, event in self._load_fixtures().values():
            try:
                scores = row["score"]["fulltime"]
                if row["fixture"]["status"]["short"] != "FT" or any(type(scores[k]) is not int or scores[k] < 0 for k in ("home", "away")):
                    continue
                events.append(dict(event, completed=True, last_update=None, scores=[
                    {"name": event["home_team"], "score": str(scores["home"])},
                    {"name": event["away_team"], "score": str(scores["away"])}]))
            except (KeyError, TypeError):
                continue
        return events, dict(self.quota)

    def fetch_events(self):
        fixtures = self._load_fixtures()
        league, season = self._discover()
        rows = self._pages("/odds", {"league": league, "season": season, "bet": 1, "page": 1})
        # Evaluate after all HTTP requests: cached fixture eligibility can expire
        # while discovery or pagination is in flight.
        raw_now = self._now() if callable(self._now) else self._now
        try:
            now = _timestamp(raw_now) if raw_now is not None else datetime.now(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            raise ApiFootballError() from None
        events = {}
        for row in rows:
            try:
                identity = row["fixture"]["id"]
                if not _positive(identity) or identity not in fixtures:
                    continue
                fixture, event = fixtures[identity]
                if (fixture.get("fixture", {}).get("status", {}).get("short") != "NS" or
                        _timestamp(event["commence_time"]) <= now):
                    continue
                if "league" in row and (not isinstance(row["league"], dict) or
                        any(type(row["league"].get(k)) is not int or row["league"][k] != v for k, v in (("id", league), ("season", season)))):
                    continue
                bookmakers = row["bookmakers"]
                if not isinstance(bookmakers, list):
                    continue
                for book in bookmakers:
                    normalized = self._book(book, fixtures[identity][1])
                    if normalized:
                        event = events.setdefault(identity, dict(fixtures[identity][1], bookmakers=[]))
                        if normalized["key"] not in {b["key"] for b in event["bookmakers"]}:
                            event["bookmakers"].append(normalized)
            except (KeyError, TypeError, AttributeError, ValueError, OverflowError):
                continue
        return list(events.values()), dict(self.quota)

    @staticmethod
    def _book(book, event):
        if not isinstance(book, dict) or not _positive(book.get("id")) or not isinstance(book.get("name"), str) or not book["name"].strip() or not isinstance(book.get("bets"), list):
            return None
        for bet in book["bets"]:
            if not isinstance(bet, dict) or type(bet.get("id")) is not int or bet["id"] != 1 or bet.get("name") != "Match Winner" or not isinstance(bet.get("values"), list):
                continue
            prices = {}
            try:
                for item in bet["values"]:
                    label, raw = item["value"], item["odd"]
                    if label not in ("Home", "Draw", "Away") or label in prices or isinstance(raw, bool) or not isinstance(raw, (str, int, float)):
                        raise ValueError()
                    price = float(raw)
                    if not math.isfinite(price) or price < 1.01:
                        raise ValueError()
                    prices[label] = price
                if len(prices) != 3:
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            key = "pinnacle" if book["name"].strip().casefold() == "pinnacle" else f"api-football:{book['id']}"
            return {"key": key, "title": book["name"], "markets": [{"key": "h2h", "outcomes": [
                {"name": name, "price": prices[label]} for label, name in
                (("Home", event["home_team"]), ("Draw", "Draw"), ("Away", event["away_team"]))]}]}
        return None
