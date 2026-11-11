"""photo-geolocator-01 entry: model-free geolocator (Python 3.13, stdlib only).

Places each street photo by its JPEG DC colour statistics: decodes the
baseline-JPEG DC coefficients with pure-Python code, builds a block-level RGB
grid, extracts colour/climate statistics, scores them against a small
hand-built table of regional priors (broad climate/vegetation regions with a
centre and a mean colour profile), and answers the geometric median of the
best-matching regions plus the world centroid. No model calls, no network
beyond the trial env, no wall clock, no randomness: fully deterministic.

Protocol (agent/v1): reads MAKEFUN_ENV_URL, GETs /photos, GETs each photo,
POSTs {"answers": [{"id", "lat", "lon"}]} to /answers. Every photo is always
answered (the world centroid is the fallback), so no photo scores the
20,015 km unanswered penalty.
"""

import json
import math
import os
import struct
import urllib.error
import urllib.request

# ---------------------------------------------------------------- JPEG DC decoder
# Baseline JPEG only (SOF0, Huffman). Returns dequantised DC per 8x8 block.
# A block whose 63 AC coefficients fill exactly ends WITHOUT an EOB marker;
# the AC loop must stop at 63 coefficients, or the next block's DC is read
# from the wrong bit (a real desync found in a practice photo).


class BitReader:
    __slots__ = ("data", "pos", "bitbuf", "bitcnt")

    def __init__(self, data, pos):
        self.data = data
        self.pos = pos
        self.bitbuf = 0
        self.bitcnt = 0

    def _fill(self):
        d = self.data
        while self.bitcnt <= 24:
            if self.pos >= len(d):
                self.bitbuf = (self.bitbuf << 8) | 0xFF
                self.bitcnt += 8
                continue
            b = d[self.pos]
            self.pos += 1
            if b == 0xFF:
                if self.pos < len(d) and d[self.pos] == 0x00:
                    self.pos += 1  # stuffed byte: skip it
                else:
                    self.bitbuf = (self.bitbuf << 8) | 0xFF
                    self.bitcnt += 8
                    continue
            self.bitbuf = (self.bitbuf << 8) | b
            self.bitcnt += 8

    def bits(self, n):
        if n == 0:
            return 0
        self._fill()
        self.bitcnt -= n
        v = (self.bitbuf >> self.bitcnt) & ((1 << n) - 1)
        self.bitbuf &= (1 << self.bitcnt) - 1 if self.bitcnt else 0
        return v

    def bit(self):
        return self.bits(1)


class Huff:
    __slots__ = ("counts", "symbols", "mincode", "maxcode", "valptr")

    def __init__(self, counts, symbols):
        self.counts = counts
        self.symbols = symbols
        self.mincode = [0] * 16
        self.maxcode = [0] * 16
        self.valptr = [0] * 16
        code = 0
        k = 0
        for l in range(16):
            if counts[l]:
                self.valptr[l] = k
                self.mincode[l] = code
                code += counts[l]
                self.maxcode[l] = code - 1
            else:
                self.maxcode[l] = -1
            code <<= 1
            k += counts[l]

    def decode(self, br):
        code = br.bit()
        for l in range(16):
            if code <= self.maxcode[l]:
                idx = self.valptr[l] + (code - self.mincode[l])
                return self.symbols[idx]
            code = (code << 1) | br.bit()
        raise ValueError("bad huffman code")


def _receive_extend(br, n):
    if n == 0:
        return 0
    v = br.bits(n)
    if v < (1 << (n - 1)):
        return v - (1 << n) + 1
    return v


def decode_dc(data):
    """JPEG bytes -> (width, height, comps, grids); grids[i] is the DC grid
    (dequantised DC per 8x8 block) of component i."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i = 2
    qtables = {}
    huff_dc = {}
    huff_ac = {}
    width = height = 0
    comps = []
    scan = None
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xD8 or m == 0xD9 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        ln = struct.unpack(">H", data[i + 2:i + 4])[0]
        seg = data[i + 4:i + 2 + ln]
        if m == 0xDB:  # DQT
            j = 0
            while j < len(seg):
                pq = seg[j] >> 4
                tq = seg[j] & 15
                j += 1
                if pq == 0:
                    qtables[tq] = list(seg[j:j + 64])
                    j += 64
                else:
                    qtables[tq] = [struct.unpack(">H", seg[j + 2 * k:j + 2 * k + 2])[0]
                                   for k in range(64)]
                    j += 128
        elif m == 0xC0:  # SOF0
            height, width = struct.unpack(">HH", seg[1:5])
            nc = seg[5]
            for k in range(nc):
                cid = seg[6 + 3 * k]
                hv = seg[7 + 3 * k]
                tq = seg[8 + 3 * k]
                comps.append({"id": cid, "h": hv >> 4, "v": hv & 15, "tq": tq})
        elif m == 0xC4:  # DHT
            j = 0
            while j < len(seg):
                tc = seg[j] >> 4
                th = seg[j] & 15
                j += 1
                counts = list(seg[j:j + 16])
                j += 16
                total = sum(counts)
                syms = list(seg[j:j + total])
                j += total
                h = Huff(counts, syms)
                if tc == 0:
                    huff_dc[th] = h
                else:
                    huff_ac[th] = h
        elif m == 0xDA:  # SOS
            ns = seg[0]
            scan = []
            for k in range(ns):
                cs = seg[1 + 2 * k]
                tdta = seg[2 + 2 * k]
                scan.append((cs, tdta >> 4, tdta & 15))
            scan_start = i + 2 + ln
            break
        i += 2 + ln
    if scan is None:
        raise ValueError("no SOS found")
    hmax = max(c["h"] for c in comps)
    vmax = max(c["v"] for c in comps)
    mcux = (width + 8 * hmax - 1) // (8 * hmax)
    mcuy = (height + 8 * vmax - 1) // (8 * vmax)
    grids = []
    for c in comps:
        bw = mcux * c["h"]
        bh = mcuy * c["v"]
        grids.append([[0] * bw for _ in range(bh)])
    comp_by_id = {c["id"]: (idx, c) for idx, c in enumerate(comps)}
    br = BitReader(data, scan_start)
    prev_dc = [0] * len(comps)
    for my in range(mcuy):
        for mx in range(mcux):
            for (cs, td, ta) in scan:
                idx, c = comp_by_id[cs]
                q = qtables[c["tq"]][0]
                for by in range(c["v"]):
                    for bx in range(c["h"]):
                        s = huff_dc[td].decode(br)
                        diff = _receive_extend(br, s)
                        prev_dc[idx] += diff
                        grids[idx][my * c["v"] + by][mx * c["h"] + bx] = prev_dc[idx] * q
                        k = 0
                        while True:
                            r = huff_ac[ta].decode(br)
                            if r == 0x00:
                                break  # EOB
                            run = r >> 4
                            size = r & 15
                            if size:
                                br.bits(size)
                            if r == 0xF0:
                                k += 16
                            else:
                                k += run + 1
                            if k >= 63:
                                break  # block full: no EOB follows
    return width, height, comps, grids


def dc_to_rgb_grid(width, height, comps, grids):
    """(rows, cols, rgb): block-level RGB grid, chroma upsampled by nearest
    neighbour to luma resolution. A block's mean pixel is ~DC/8 + 128."""
    hmax = max(c["h"] for c in comps)
    vmax = max(c["v"] for c in comps)
    mcux = (width + 8 * hmax - 1) // (8 * hmax)
    mcuy = (height + 8 * vmax - 1) // (8 * vmax)
    rows = mcuy * vmax
    cols = mcux * hmax
    yi = 0
    cbi = 1 if len(comps) > 1 else 0
    cri = 2 if len(comps) > 2 else 0

    def comp_val(g, r, c):
        gh = len(g)
        gw = len(g[0])
        rr = min(r * gh // rows, gh - 1)
        cc = min(c * gw // cols, gw - 1)
        return g[rr][cc] / 8.0

    yg = grids[yi]
    cbg = grids[cbi]
    crg = grids[cri]
    rgb = []
    for r in range(rows):
        row = []
        for c in range(cols):
            y = comp_val(yg, r, c) + 128
            cb = comp_val(cbg, r, c) + 128
            cr = comp_val(crg, r, c) + 128
            R = y + 1.402 * (cr - 128)
            G = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
            B = y + 1.772 * (cb - 128)
            row.append((max(0, min(255, R)), max(0, min(255, G)), max(0, min(255, B))))
        rgb.append(row)
    return rows, cols, rgb


# ---------------------------------------------------------------- features
def extract(rgb):
    """Colour/climate statistics of the block RGB grid. Only the features the
    anchor table uses are returned, in FEATS order."""
    rows = len(rgb)
    cols = len(rgb[0])

    def stats(cells):
        n = len(cells)
        return (sum(c[0] for c in cells) / n,
                sum(c[1] for c in cells) / n,
                sum(c[2] for c in cells) / n)

    allcells = [c for row in rgb for c in row]
    mr, mg, mb = stats(allcells)
    ys = [0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] for c in allcells]
    my = sum(ys) / len(ys)
    vy = sum((y - my) ** 2 for y in ys) / len(ys)
    sy = math.sqrt(vy)

    def band(r0, r1):
        cells = [c for row in rgb[int(rows * r0):max(int(rows * r1), int(rows * r0) + 1)]
                 for c in row]
        return stats(cells)

    sky = band(0.0, 0.18)
    mid = band(0.30, 0.65)
    gnd = band(0.72, 1.0)
    n = len(allcells)
    blue = white = gray = yellow = 0
    for (R, G, B) in allcells:
        if B > R + 20 and B > G + 8:
            blue += 1
        if R > 195 and G > 195 and B > 195 and max(R, G, B) - min(R, G, B) < 25:
            white += 1
        if max(R, G, B) - min(R, G, B) < 14 and 50 < (R + G + B) / 3 < 175:
            gray += 1
        if R > 140 and G > 105 and B < 110 and R - B > 50:
            yellow += 1
    ly = sum(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
             for row in rgb for c in row[:cols // 2]) / (rows * cols // 2)
    ry = sum(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
             for row in rgb for c in row[cols // 2:]) / (rows * cols // 2)
    return {
        "bluesky": blue / n,
        "skyBlue": sky[2] - sky[0],
        "skySat": max(sky) - min(sky),
        "midB": mid[2],
        "B": mb,
        "midGreen": mid[1] - mid[0],
        "midG": mid[1],
        "gray": gray / n,
        "gndB": gnd[2],
        "yellow": yellow / n,
        "white": white / n,
        "gndG": gnd[1],
        "gndR": gnd[0],
        "gndRed": gnd[0] - gnd[1],
        "lrgrad": ly - ry,
        "Ysd": sy,
    }


# ---------------------------------------------------------------- regional priors
# Hand-built anchor table. Each anchor is a broad climate/vegetation region
# (lat, lon = regional centre) with the mean colour/climate profile of street
# photos from that region, calibrated on the kit's 75 public practice photos.
# Aggregated regions, not individual photos or places: no photo table.
FEATS = ["bluesky", "skyBlue", "skySat", "midB", "B", "midGreen", "midG",
         "gray", "gndB", "yellow", "white", "gndG", "gndR", "gndRed",
         "lrgrad", "Ysd"]
# (mean, std) of each feature across the practice photos, for standardising.
STD = [(0.314, 0.255), (53.074, 46.266), (54.184, 45.14), (128.298, 43.44),
       (129.422, 27.526), (16.863, 20.781), (124.34, 33.521), (0.156, 0.136),
       (93.788, 44.216), (0.004, 0.015), (0.074, 0.1), (99.676, 47.782),
       (94.938, 48.582), (-4.738, 14.744), (1.372, 32.331), (58.526, 12.834)]
# Practice photos' mean place: the stabilising prior when the signal is weak.
CENTROID = (22.17, 10.51)
# (lat, lon, [profile in FEATS order])
ANCHORS = [
    (-37.7, -67.8, [0.3, 60.5, 60.5, 149.6, 155.7, 38.5, 143.6, 0.2, 141.6, 0.0, 0.1, 170.0, 175.1, 5.1, -8.0, 40.6]),
    (-44.2, 169.9, [0.3, 64.9, 64.9, 76.3, 102.6, 11.3, 76.1, 0.1, 74.2, 0.0, 0.1, 76.8, 69.5, -7.3, -23.5, 70.2]),
    (-25.1, -67.4, [0.3, 84.8, 84.8, 118.0, 140.4, -0.2, 126.7, 0.1, 133.7, 0.0, 0.0, 144.6, 141.8, -2.8, 9.0, 42.0]),
    (-19.3, -45.4, [0.1, 34.2, 34.3, 110.5, 118.4, 11.9, 110.3, 0.2, 71.1, 0.0, 0.1, 79.8, 77.4, -2.3, -4.2, 64.0]),
    (-13.5, 40.3, [0.5, 137.8, 137.8, 178.6, 165.3, 66.2, 159.1, 0.1, 120.8, 0.0, 0.0, 142.8, 133.4, -9.4, -9.3, 47.6]),
    (-15.6, 147.9, [0.3, 72.0, 72.0, 97.8, 112.2, 15.0, 111.9, 0.0, 69.5, 0.0, 0.0, 88.0, 84.7, -3.3, -9.6, 58.0]),
    (22.4, -97.0, [0.3, 27.7, 29.9, 158.4, 129.5, 11.8, 151.9, 0.2, 68.5, 0.0, 0.1, 68.6, 67.0, -1.6, -30.0, 63.6]),
    (7.9, -6.6, [0.3, 61.3, 61.3, 125.1, 118.1, 26.6, 112.5, 0.1, 61.0, 0.0, 0.0, 66.2, 58.6, -7.6, 16.3, 63.5]),
    (8.2, 34.6, [0.4, 47.1, 51.0, 106.7, 133.5, 0.2, 112.7, 0.0, 139.8, 0.0, 0.1, 141.6, 145.3, 3.7, -11.8, 56.7]),
    (15.1, 87.6, [0.2, 24.4, 29.6, 107.4, 115.1, 10.3, 110.1, 0.2, 100.6, 0.0, 0.1, 105.5, 102.5, -3.0, 4.3, 58.8]),
    (50.2, -124.1, [0.3, 41.0, 42.5, 123.8, 99.5, 5.0, 112.9, 0.1, 48.2, 0.0, 0.0, 42.7, 41.1, -1.6, 35.1, 53.8]),
    (42.6, -94.3, [0.4, 62.9, 64.1, 144.7, 149.5, 24.8, 138.1, 0.1, 136.8, 0.0, 0.0, 144.6, 138.9, -5.7, 13.2, 47.1]),
    (46.2, 33.4, [0.3, 45.5, 45.9, 139.1, 137.5, 12.8, 137.6, 0.2, 99.9, 0.0, 0.1, 100.8, 92.9, -7.9, 8.2, 65.0]),
    (46.8, 71.2, [0.2, 22.2, 22.6, 106.5, 115.0, 7.5, 107.2, 0.3, 73.3, 0.0, 0.1, 81.0, 79.0, -2.0, 3.9, 63.4]),
    (40.5, 133.8, [0.3, 44.5, 44.9, 120.8, 116.1, 10.0, 118.2, 0.2, 78.0, 0.0, 0.1, 76.3, 71.9, -4.4, -19.5, 64.9]),
    (62.2, -136.5, [0.7, 144.3, 144.3, 213.8, 178.6, 50.6, 168.9, 0.0, 95.8, 0.0, 0.0, 106.6, 111.4, 4.8, 10.0, 53.3]),
    (67.0, 22.5, [0.5, 63.5, 64.6, 122.5, 122.6, 17.4, 110.2, 0.1, 77.7, 0.0, 0.1, 79.2, 64.7, -14.6, 14.6, 54.5]),
]

TOP_K = 3          # best-matching anchors kept
CENTROID_W = 2     # centroid copies in the median (stabiliser)


def haversine(lat1, lon1, lat2, lon2):
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371 * math.asin(min(1, math.sqrt(a)))


def geometric_median(points):
    """Weiszfeld iteration, deterministic: fixed start (mean) and 50 rounds."""
    lat = sum(p[0] for p in points) / len(points)
    lon = sum(p[1] for p in points) / len(points)
    for _ in range(50):
        nla = nlo = den = 0.0
        for (la, lo) in points:
            d = haversine(lat, lon, la, lo)
            if d < 1e-9:
                return la, lo
            w = 1.0 / d
            nla += la * w
            nlo += lo * w
            den += w
        nlat = nla / den
        nlon = nlo / den
        if abs(nlat - lat) < 1e-6 and abs(nlon - lon) < 1e-6:
            break
        lat, lon = nlat, nlon
    return lat, lon


def predict(feats):
    """(lat, lon) for one photo's feature dict."""
    scored = []
    for (la, lo, prof) in ANCHORS:
        s = 0.0
        for i, f in enumerate(FEATS):
            m, sd = STD[i]
            s += (((feats[f] - m) / sd) - ((prof[i] - m) / sd)) ** 2
        scored.append((math.sqrt(s), la, lo))
    scored.sort(key=lambda t: t[0])
    pts = [(la, lo) for (_, la, lo) in scored[:TOP_K]]
    pts += [CENTROID] * CENTROID_W
    lat, lon = geometric_median(pts)
    lat = max(-90.0, min(90.0, lat))
    lon = max(-180.0, min(180.0, lon))
    return lat, lon


def locate(jpeg_bytes):
    """(lat, lon) for one photo's JPEG bytes; CENTROID when it cannot be read."""
    try:
        w, h, comps, grids = decode_dc(jpeg_bytes)
        _, _, rgb = dc_to_rgb_grid(w, h, comps, grids)
        return predict(extract(rgb))
    except Exception:
        return CENTROID


# ---------------------------------------------------------------- env protocol
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http(method, url, body=None):
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json",
                                          "Accept": "application/json"})
    try:
        with _OPENER.open(req, timeout=120) as res:
            return res.status, res.read()
    except urllib.error.HTTPError as err:
        try:
            return err.code, err.read()
        finally:
            err.close()


def main():
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        raise SystemExit("MAKEFUN_ENV_URL is not set")
    status, raw = _http("GET", env_url + "/photos")
    if status != 200:
        raise SystemExit("GET /photos failed: %d" % status)
    photos = json.loads(raw.decode("utf-8"))["photos"]
    answers = []
    for photo in photos:
        if photo.get("answered"):
            continue
        pid = photo["id"]
        try:
            _, data = _http("GET", env_url + "/photos/" + pid + ".jpg")
            lat, lon = locate(data)
        except Exception:
            lat, lon = CENTROID
        answers.append({"id": pid, "lat": round(lat, 6), "lon": round(lon, 6)})
    if not answers:
        return
    body = json.dumps({"answers": answers}, separators=(",", ":")).encode("utf-8")
    status, _ = _http("POST", env_url + "/answers", body)
    if status != 200:
        # Retry one by one so a single bad answer cannot sink the batch.
        for a in answers:
            b = json.dumps({"answers": [a]}, separators=(",", ":")).encode("utf-8")
            _http("POST", env_url + "/answers", b)


if __name__ == "__main__":
    main()
