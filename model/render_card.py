"""POTD card renderer — fills the auto template and screenshots it with Playwright.

Photo modes (auto-selected by make_potd_card, passed in spec):
  action      - full-bleed action shot (manual upload or auto-fetched MLB 'hero' image)
  silo        - transparent player cutout on brand gradient (auto-fetched, Props.Cash-style)
  placeholder - branded initials medallion (never blocks the run)

Includes runtime JS passes: overflow shrink-fit on every text node, and measured
centering of the market/hero/logo lockup (same treatment as the approved Witt card).
"""
import base64, json, pathlib, sys

TPL = pathlib.Path("/home/claude/mlb_model/potd_auto_template.html")
LOGO_B64 = pathlib.Path("/home/claude/mlb_model/output/logo_b64.txt")

PHOTO_BLOCKS = {
    "action": '''<div class="photo-band">
    <img class="hero-blur" src="__PHOTO_SRC__" alt="" />
    <img class="hero-img" src="__PHOTO_SRC__" alt="player" />
    <div class="glow-photo"></div>
    <div class="wash-sides"></div>
    <div class="wash-bottom"></div>
    <div class="wash-top"></div>
  </div>''',
    "silo": '''<div class="photo-band">
    <div class="glow-photo"></div>
    <img class="silo-img" src="__PHOTO_SRC__" alt="player" />
    <div class="wash-bottom"></div>
  </div>''',
    "placeholder": '''<div class="photo-band">
    <div class="glow-photo"></div>
    <div class="medallion"><span>__INITIALS__</span></div>
    <div class="wash-bottom"></div>
  </div>''',
}

TILE = '<div class="tile c{c} r{r}"><div class="val">{val}</div><div class="lab">{lab}</div></div>'

FIT_JS = """
() => {
  // shrink any nowrap text that overflows its box
  const fit = (el, min) => {
    let fs = parseFloat(getComputedStyle(el).fontSize);
    while (el.scrollWidth > el.clientWidth && fs > min) { fs -= 1; el.style.fontSize = fs + 'px'; }
  };
  document.querySelectorAll('.tile .val').forEach(e => fit(e, 26));
  document.querySelectorAll('.tile .lab').forEach(e => fit(e, 12));
  ['.name', '.matchup', '.season', '.support', '.kickerline'].forEach(sel => {
    const e = document.querySelector(sel); if (e) fit(e, 14);
  });
  // measured centering of the market/hero/logo lockup
  const m = document.querySelector('.market'), h = document.querySelector('.hero'),
        lg = document.querySelector('.logo-badge');
  const mw = m.getBoundingClientRect().width, hw = h.getBoundingClientRect().width;
  const block = Math.max(mw, hw), GAP = 64, LOGO = 176, OPT = -14;
  const left = Math.round((1080 - (block + GAP + LOGO)) / 2) + OPT;
  h.style.left = left + 'px';
  m.style.left = (left + 4) + 'px';
  lg.style.left = Math.round(left + block + GAP) + 'px';
  return {block: Math.round(block), left};
}
"""


def build_html(spec: dict) -> str:
    html = TPL.read_text()
    mode = spec.get("photo_mode", "placeholder")
    block = PHOTO_BLOCKS[mode]
    if mode == "placeholder":
        block = block.replace("__INITIALS__", spec.get("initials", "?"))
    else:
        src = spec["photo_src"]  # data URI already
        block = block.replace("__PHOTO_SRC__", src)
    tiles = []
    pos = [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]
    for (c, r), t in zip(pos, spec["tiles"][:6]):
        tiles.append(TILE.format(c=c, r=r, val=t["val"], lab=t["lab"]))
    if mode == "action":  # over a full-bleed photo the kicker sits left, clear of the subject
        html = html.replace('class="kickerline"', 'class="kickerline kicker-action"')
    html = (html
            .replace("__PHOTO_BLOCK__", block)
            .replace("__KICKER_DATE__", spec["kicker_date"])
            .replace("__NAME__", spec["name"].upper())
            .replace("__MATCHUP__", spec["matchup"])
            .replace("__SEASON__", spec["season_line"])
            .replace("__MARKET__", spec["market"].upper())
            .replace("__HERO__", spec["hero_line"].upper())
            .replace("__LOGO__", LOGO_B64.read_text().strip())
            .replace("__TILES__", "\n    ".join(tiles))
            .replace("__SUPPORT__", spec["support"]))
    return html


def render(spec: dict, out_png: str, out_html: str | None = None) -> dict:
    from playwright.sync_api import sync_playwright
    html = build_html(spec)
    tmp = pathlib.Path("/tmp/potd_card_render.html")
    tmp.write_text(html)
    if out_html:
        pathlib.Path(out_html).write_text(html)
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1080, "height": 1350}, device_scale_factor=2)
        page.goto(f"file://{tmp}")
        page.wait_for_timeout(400)
        info = page.evaluate(FIT_JS)
        page.wait_for_timeout(150)
        page.locator(".graphic").screenshot(path=out_png)
        b.close()
    return info


def img_to_data_uri(path: str) -> str:
    p = pathlib.Path(path)
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    print(render(spec, sys.argv[2]))
