"""Capture README screenshots from a RUNNING local VOLSCAPE (http://127.0.0.1:8501) with headless Edge/Chrome
over the DevTools protocol. Real captures of the real app; nothing is mocked.

    python scripts/capture_screenshots.py [--live]

--live also connects the LIVE workspace (opens public Binance/Yahoo connections) and captures it.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
URL = "http://127.0.0.1:8501/"
W, H = 1600, 1000
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium"]


class CDP:
    def __init__(self, ws):
        self.ws, self.i = ws, 0

    async def call(self, method, **params):
        self.i += 1
        mid = self.i
        await self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        return r.get("result", {}).get("value")

    async def wait_for(self, expr, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if await self.js(expr):
                return True
            await asyncio.sleep(0.5)
        raise TimeoutError(expr)

    async def shot(self, name, settle=2.5):
        await asyncio.sleep(settle)
        await self.js("window.dispatchEvent(new Event('resize')); 1")   # grids re-measure after a tab reveal
        await asyncio.sleep(1.2)
        n = await self.js("document.querySelectorAll('[data-testid=stDataFrame]').length") or 0
        for i in range(n):                  # canvas tables paint lazily: wait for each visible one
            for _ in range(30):
                vis = await self.js(f"""(() => {{ const r=document.querySelectorAll('[data-testid=stDataFrame]')[{i}]
                    .getBoundingClientRect(); return r.bottom > 0 && r.top < {H}; }})()""")
                if not vis or await self.painted(i):
                    break
                await asyncio.sleep(0.3)
        data = (await self.call("Page.captureScreenshot", format="jpeg", quality=88))["data"]
        (OUT / name).write_bytes(base64.b64decode(data))
        print("captured", name)

    async def pick_option(self, combo_label, option_text):
        """Open a Streamlit selectbox by its aria label and click an option by text."""
        ok = await self.js(f"""(() => {{ const c=[...document.querySelectorAll('[role=combobox]')]
            .find(x=>(x.getAttribute('aria-label')||'').includes({json.dumps(combo_label)}));
            if(!c) return false; c.click(); c.focus(); return true; }})()""")
        if not ok:
            raise RuntimeError(f"no combobox {combo_label!r}")
        await asyncio.sleep(0.8)
        ok = await self.js(f"""(() => {{ const o=[...document.querySelectorAll('[role=option]')]
            .find(x=>x.innerText.trim()==={json.dumps(option_text)}); if(!o) return false; o.click(); return true; }})()""")
        if not ok:
            raise RuntimeError(f"no option {option_text!r}")

    async def painted(self, testid_index=0, min_bright=0.004):
        """True if the n-th dataframe canvas region contains light (text) pixels in a fresh capture."""
        box = await self.js(f"""(() => {{ const d=document.querySelectorAll('[data-testid=stDataFrame]')[{testid_index}];
            if(!d) return null; const r=d.getBoundingClientRect(); return [r.x, r.y, r.width, r.height]; }})()""")
        if not box or box[2] < 20 or box[3] < 20:
            return False
        box = [box[0], max(box[1], 0), box[2], min(box[1] + box[3], H) - max(box[1], 0)]
        if box[3] < 20:
            return False
        from io import BytesIO
        from PIL import Image
        data = (await self.call("Page.captureScreenshot", format="png"))["data"]
        im = Image.open(BytesIO(base64.b64decode(data))).convert("L")
        x, y, w, h = (int(v) for v in box)
        px = im.crop((x, y, x + w, y + h)).getdata()
        return sum(1 for p in px if p > 120) / max(1, len(px)) > min_bright

    async def click_text(self, text, selector="button, [role=tab], label, p"):
        ok = await self.js(f"""(() => {{ const els=[...document.querySelectorAll({json.dumps(selector)})];
            const e=els.find(x=>x.innerText && x.innerText.trim().toLowerCase()==={json.dumps(text.lower())});
            if(!e) return false; e.scrollIntoView({{block:'start'}}); e.click(); return true; }})()""")
        if not ok:
            raise RuntimeError(f"no element with text {text!r}")

    async def scroll_to_text(self, text, offset=-70):
        await self.js(f"""(() => {{ const e=[...document.querySelectorAll('div,p,span,h1,h2,h3')]
            .find(x=>x.childElementCount===0 && x.innerText && x.innerText.includes({json.dumps(text)}));
            if(e){{ const r=e.getBoundingClientRect(); const sc=document.querySelector('[data-testid="stMain"]')||document.scrollingElement;
            (document.scrollingElement).scrollBy(0, r.top+({offset})); sc.scrollBy && sc.scrollBy(0, r.top+({offset})); }} }})()""")


async def run(live: bool):
    exe = next((b for b in BROWSERS if os.path.exists(b)), None)
    if exe is None:
        sys.exit("No Edge/Chrome found")
    prof = tempfile.mkdtemp(prefix="volscape-shot-")
    proc = subprocess.Popen([exe, "--headless=new", f"--window-size={W},{H}", "--remote-debugging-port=9233",
                             f"--user-data-dir={prof}", "--hide-scrollbars", "--force-device-scale-factor=1",
                             "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9233/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        async with connect(page["webSocketDebuggerUrl"], max_size=2**26) as ws:
            c = CDP(ws)
            await c.call("Page.enable")
            await c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
            await c.call("Page.navigate", url=URL)
            await c.wait_for("document.querySelectorAll('.js-plotly-plot').length >= 3", 120)
            await asyncio.sleep(4)
            await c.shot("workstation_surface.jpg")
            await c.click_text("Arbitrage Diagnostics")
            await asyncio.sleep(2)
            await c.scroll_to_text("Risk-neutral density waterfall", -40)
            await c.shot("density_waterfall.jpg", 3)
            if live:
                await c.js("window.scrollTo(0,0)")
                await c.pick_option("Dataset", "live · deribit BTC")
                await c.wait_for("document.body.innerText.toLowerCase().includes('live snapshot')", 90)
                await asyncio.sleep(6)
                await c.js("window.scrollTo(0,0)")
                await c.shot("live_deribit_surface.jpg", 2)
                await c.click_text("LIVE", "label, p")
                await c.wait_for("document.body.innerText.includes('Not connected') || document.body.innerText.includes('Disconnect') || document.body.innerText.includes('DISCONNECT')", 30)
                await asyncio.sleep(2)
                if await c.js("document.body.innerText.includes('Not connected')"):
                    await c.js("[...document.querySelectorAll('button')].find(b=>b.innerText.trim().toLowerCase()==='connect').click(); 1")
                await c.wait_for("[...document.querySelectorAll('iframe')].some(f => (f.srcdoc || '').includes('landscape'))", 60)
                await asyncio.sleep(30)          # 24h history + book samples accumulate from the real feed
                await c.js("window.scrollTo(0, 230)")
                await c.shot("live_crypto_panel.jpg", 2)
                await c.js("window.scrollTo(0, 620)")
                await c.shot("live_orderbook_3d.jpg", 1)
                try:
                    await c.click_text("Disconnect", "button, p")
                except RuntimeError:
                    pass
                await asyncio.sleep(2)
                await c.click_text("stocks & indices · delayed", "label, p")
                await asyncio.sleep(2)
                await c.click_text("Load delayed day chart (connects to Yahoo Finance)", "button, p")
                await asyncio.sleep(4)
                await c.pick_option("Stock / index", "Reliance Industries · INDIA")
                await c.wait_for("document.querySelectorAll('.js-plotly-plot').length >= 2", 60)
                await c.js("const p=[...document.querySelectorAll('.js-plotly-plot')].pop(); p.scrollIntoView({block:'center'}); 1")
                await c.shot("stock_volume_3d.jpg", 6)
    finally:
        proc.terminate()


if __name__ == "__main__":
    asyncio.run(run("--live" in sys.argv))
