# app.py
# Streamlit app: Dropshipping Product Hunter + Ad Watch & Creative Studio
# What it does (free stack):
# 1) Product hunting: Google Trends momentum score for keywords; quick links to official ad libraries to see live ads.
# 2) Sales estimator: projections from known order counts (AliExpress, Shopify, Amazon) + margin calculator. No scraping of restricted sites.
# 3) Ad Watch: one-click deep links to Meta/TikTok/Google/Snap/Pinterest ad libraries per keyword.
# 4) Creative Studio: auto-generate ad scripts/captions; create ad images with overlays; assemble simple MP4 videos from images using MoviePy.
#
# Run locally: `pip install streamlit pytrends pandas numpy pillow moviepy requests`
# Start: `streamlit run app.py`

import io
import math
import textwrap
import urllib.parse as up
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from pytrends.request import TrendReq

st.set_page_config(page_title="Dropship Hunter + Creative Studio", page_icon="🛒", layout="wide")

# -------------------------- Utils --------------------------

def _safe_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default

@st.cache_data(show_spinner=False)
def trends_series(keyword: str, geo: str = "US", timeframe: str = "today 3-m"):
    try:
        py = TrendReq(hl="en-US", tz=360)
        py.build_payload([keyword], timeframe=timeframe, geo=geo)
        df = py.interest_over_time()
        return df[keyword] if not df.empty else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)

def momentum(s: pd.Series) -> float:
    if s is None or s.empty or len(s) < 8:
        return 0.0
    y = s.values.astype(float)
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    recent = y[-8:]; earlier = y[:8] if len(y)>=16 else y[:max(1,len(y)//3)]
    growth = (recent.mean() - (earlier.mean() if len(earlier) else 0))
    vol = np.std(np.diff(y)) + 1e-6
    score = 50 + 400*slope/(np.max(y)+1e-6) + 2*growth + min(30, 10/vol)
    return float(max(0, min(100, score)))


def ad_links(q: str, country: str = "US", lang: str = "en") -> dict:
    query = up.quote(q)
    return {
        "Meta (FB/IG)": f"https://www.facebook.com/ads/library/?active_status=all&ad_type=all&country={country}&q={query}",
        "TikTok Creative Center": f"https://ads.tiktok.com/business/creativecenter/inspiration/topads/pc/{lang}?search_word={query}",
        "Google Ads Transparency": f"https://adstransparency.google.com/?search={query}",
        "Snap Ads Library": f"https://www.snap.com/en-US/safety/snap-ads-library?search={query}",
        "Pinterest Trends": f"https://trends.pinterest.com/search/{country}/{query}"
    }


def margin_calc(cost, price, ship=0.0, fees_pct=0.05):
    try:
        cost=float(cost or 0); price=float(price or 0); ship=float(ship or 0); fees=float(fees_pct or 0)
        gross = price - cost - ship - price*fees
        pct = (gross/price*100) if price else 0
        return round(gross,2), round(pct,1)
    except Exception:
        return 0.0, 0.0


def revenue_projection(orders_per_month: int, price: float):
    orders=_safe_int(orders_per_month,0)
    rev = orders*float(price or 0)
    return int(orders), round(rev,2)

# -------------------------- UI --------------------------

st.title("🛒 Dropshipping Product Hunter + Ad Watch & Creative Studio")
st.caption("Find promising products, see where ads are running (via official libraries), estimate sales, and generate creatives — all free to run.")

tabs = st.tabs(["Product Hunt","Sales Estimator","Ad Watch","Creative Studio"])  # 4 tabs

# ------------------- Product Hunt -------------------
with tabs[0]:
    colL, colR = st.columns([2,1])
    with colL:
        st.subheader("Enter keywords / product ideas")
        raw = st.text_area("One per line (e.g., 'posture corrector', 'mini blender', 'magnetic lashes')", height=120)
        uploaded = st.file_uploader("…or upload CSV with 'keyword' column", type=["csv"], key="huntcsv")
        kws = []
        if raw.strip():
            kws += [k.strip() for k in raw.splitlines() if k.strip()]
        if uploaded:
            try:
                dfu = pd.read_csv(uploaded)
                kws += [str(x) for x in dfu.get('keyword').dropna().tolist()]
            except Exception:
                st.warning("Could not read CSV. Ensure it has a 'keyword' column.")
        kws = list(dict.fromkeys(kws))

    with colR:
        region = st.selectbox("Region (Google Trends)", ["US","GB","CA","AU","PK","IN","SA","AE","DE","FR"], index=0)
        timeframe = st.selectbox("Timeframe", ["today 3-m","today 12-m","now 7-d","now 1-d"], index=0)
        st.info("Momentum is computed from slope + recent growth + consistency. Use it to shortlist ideas.")

    if kws:
        rows = []
        prog = st.progress(0.0)
        for i, kw in enumerate(kws, start=1):
            s = trends_series(kw, geo=region, timeframe=timeframe)
            m = momentum(s)
            latest = float(s.iloc[-1]) if not s.empty else 0.0
            rows.append({"keyword": kw, "momentum": round(m,1), "latest_interest": round(latest,1)})
            prog.progress(i/len(kws))
        prog.empty()
        df = pd.DataFrame(rows).sort_values(["momentum","latest_interest"], ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download shortlist CSV", df.to_csv(index=False).encode("utf-8"), file_name="shortlist.csv")
    else:
        st.info("Add some keywords to start.")

# ------------------- Sales Estimator -------------------
with tabs[1]:
    st.subheader("Sales & Margin Estimator (input what you know)")
    st.caption("Enter any known signals (monthly orders from AliExpress listing, Etsy, Amazon reviews velocity, Shopify store analytics, etc.). This tool models projections; it does not scrape restricted data.")
    cols = st.columns(4)
    src = cols[0].selectbox("Source of orders", ["AliExpress","Amazon","Shopify","Etsy","Other"], index=0)
    orders = cols[1].number_input("Monthly orders (est.)", min_value=0, step=10, value=200)
    price  = cols[2].number_input("Sell price (USD)", min_value=0.0, step=0.5, value=24.99)
    cost   = cols[3].number_input("Product cost (USD)", min_value=0.0, step=0.5, value=6.00)
    ship_fees = st.columns(2)
    ship = ship_fees[0].number_input("Shipping per order", min_value=0.0, step=0.5, value=0.0)
    fees = ship_fees[1].slider("Platform/processing fees %", 0.00, 0.20, 0.05, 0.01)

    gross, pct = margin_calc(cost, price, ship, fees)
    mo_orders, mo_rev = revenue_projection(orders, price)
    st.metric("Estimated gross per order", f"${gross} ({pct}%)")
    st.metric("Monthly revenue (est.)", f"${mo_rev}")
    st.metric("Monthly gross profit (est.)", f"${round(mo_orders * gross, 2)}")

    st.markdown("**Tip:** If you have only review counts on Amazon, use rule-of-thumb conversions (e.g., ~2–5% buyers leave reviews) to back into sales, then plug here.")

# ------------------- Ad Watch -------------------
with tabs[2]:
    st.subheader("Check which platforms have live ads for your keyword")
    country = st.selectbox("Country for libraries", ["US","GB","CA","AU","PK","IN","SA","AE","DE","FR"], index=0, key="country")
    lang = st.selectbox("Language (TikTok center)", ["en","ar","fr","de","es","pt","hi"], index=0)
    q = st.text_input("Keyword / brand / product name", placeholder="e.g., posture corrector")
    if q:
        links = ad_links(q, country=country, lang=lang)
        cols = st.columns(3)
        i=0
        for name, url in links.items():
            if i % 3 == 0 and i>0:
                cols = st.columns(3)
            with cols[i%3]:
                st.link_button(name, url)
            i+=1
        st.info("These open official public ad libraries in a new tab. You can verify if ads are currently active and see creatives and spend ranges where available.")
    else:
        st.info("Enter a keyword to generate the links.")

# ------------------- Creative Studio -------------------
with tabs[3]:
    st.subheader("Creative Studio — scripts, images, and quick videos (free)")
    st.markdown("#### A) Auto-generate ad copy & scripts")
    prod = st.text_input("Product name", "Magnetic Lashes")
    usp = st.text_area("Key benefits / USP (one per line)", "No glue needed\nWaterproof\nReusable up to 30 wears")
    target = st.text_input("Audience", "Busy moms, students")
    angle = st.selectbox("Angle", ["Problem → Solution","Before / After","Social proof","How-to demo"], index=0)
    if st.button("Generate scripts & captions"):
        bullets = [x.strip() for x in usp.splitlines() if x.strip()]
        hook = {
            "Problem → Solution": f"Tired of messy glue? Meet {prod} — on in seconds.",
            "Before / After": f"From bare to bold in 5 seconds — {prod}.",
            "Social proof": f"Join 50,000+ who switched to {prod}.",
            "How-to demo": f"Watch how {prod} snaps on — no glue."
        }[angle]
        lines = [f"Hook: {hook}", "Benefits:"] + [f"• {b}" for b in bullets] + ["CTA: Tap Shop Now"]
        st.code("\n".join(lines), language="markdown")

    st.markdown("#### B) Create ad images with overlays")
    img_files = st.file_uploader("Upload 1–5 product images (PNG/JPG)", type=["png","jpg","jpeg"], accept_multiple_files=True)
    hl = st.text_input("Headline", "No-Glue Magnetic Lashes")
    price_tag = st.text_input("Price badge (optional)", "$19.99 Today")
    cta = st.text_input("CTA", "Shop Now →")
    if st.button("Make image creatives", disabled=not img_files):
        out_zip_bytes = io.BytesIO()
        import zipfile
        with zipfile.ZipFile(out_zip_bytes, 'w', zipfile.ZIP_DEFLATED) as zf:
            for idx, f in enumerate(img_files, start=1):
                img = Image.open(f).convert("RGBA").resize((1080,1080))
                draw = ImageDraw.Draw(img)
                # simple semi-transparent bar
                bar_h = 260
                overlay = Image.new('RGBA', (1080, bar_h), (0,0,0,140))
                img.alpha_composite(overlay, (0, 1080-bar_h))
                # fonts: rely on PIL default; for better look, user can add .ttf path
                draw.text((40, 860), hl, fill=(255,255,255,255))
                draw.text((40, 940), price_tag, fill=(144,238,144,255))
                draw.text((40, 1000), cta, fill=(255,255,255,255))
                buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=92)
                zf.writestr(f"creative_{idx}.jpg", buf.getvalue())
        st.download_button("Download image pack (ZIP)", data=out_zip_bytes.getvalue(), file_name="image_creatives.zip")

    st.markdown("#### C) Assemble quick video from images (MP4)")
    vid_imgs = st.file_uploader("Upload 2–8 images", type=["png","jpg","jpeg"], accept_multiple_files=True, key="vidimgs")
    per_scene = st.slider("Seconds per scene", 1.0, 5.0, 2.0, 0.5)
    audio = st.file_uploader("Optional background audio (MP3/M4A)", type=["mp3","m4a"], key="audio")
    if st.button("Render video", disabled=not vid_imgs):
        clips = []
        for f in vid_imgs:
            img = Image.open(f).convert("RGB").resize((1080,1080))
            buf = io.BytesIO(); img.save(buf, format="JPEG", quality=92)
            clip = ImageClip(buf.getvalue()).set_duration(per_scene)
            clips.append(clip)
        video = concatenate_videoclips(clips, method="compose")
        if audio is not None:
            audioclip = AudioFileClip(audio.name)
            video = video.set_audio(audioclip.set_duration(video.duration))
        out = io.BytesIO()
        video.write_videofile("out.mp4", fps=30, codec="libx264", audio_codec="aac")
        with open("out.mp4","rb") as f:
            st.download_button("Download video (MP4)", data=f.read(), file_name="creative.mp4", mime="video/mp4")
        for c in clips: c.close()

st.markdown("---")
st.caption("Notes: This app avoids scraping restricted sources; use official ad libraries to verify live ads. Provide your own royalty‑free audio for videos. All processing happens locally while the app runs.")
