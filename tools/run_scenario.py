import json
import sys
import time
from pathlib import Path

import httpx


def main(path: str) -> None:
    scenario = json.loads(Path(path).read_text(encoding="utf-8"))
    checks = scenario.get("checks", [])
    if not checks:
        raise RuntimeError("No checks in scenario.json")

    with httpx.Client(timeout=10.0) as client:
        for c in checks:
            name = c["name"]
            method = c["method"].upper()
            url = c["url"]
            expect = int(c.get("expect_status", 200))
            payload = c.get("json")

            last_status = None
            for _ in range(30):
                try:
                    if method == "GET":
                        r = client.get(url)
                    elif method == "POST":
                        r = client.post(url, json=payload)
                    else:
                        raise RuntimeError(f"Unsupported method: {method}")
                    last_status = r.status_code
                    if last_status == expect:
                        break
                except Exception:
                    pass
                time.sleep(1)

            if last_status != expect:
                raise RuntimeError(f"[{name}] FAILED: expected {expect}, got {last_status}")

            print(f"[{name}] OK ({expect})")


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "scenario.json"
    main(p)