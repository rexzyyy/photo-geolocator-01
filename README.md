# photo-geolocator-01 — model-free geolocator

A model-free entry for the make.fun bounty **"Find where a photo was taken"**
(`photo-geolocator-01`). It places each street photo by its pixels alone:
no model calls, no network beyond the trial env, no trained weights,
no photo table.

## How it works

1. **JPEG DC decode (pure Python, stdlib only).** Each photo is a baseline
   JPEG; the bot decodes only the DC coefficients (the mean colour of every
   8×8 block) with a small Huffman + bit-reader written from the JPEG spec.
   A block whose 63 AC coefficients fill exactly ends without an EOB marker,
   and the decoder stops at 63 coefficients so the next block's DC is read
   from the right bit.
2. **Colour/climate statistics.** From the block-level RGB grid it extracts
   sky/mid/ground band colours, sky blueness, vegetation, concrete, snow and
   brightness statistics.
3. **Regional priors.** The statistics are scored against a small hand-built
   table of 17 broad climate/vegetation regions (a centre + a mean colour
   profile each, calibrated on the kit's 75 public practice photos).
4. **Geometric median.** The answer is the spherical geometric median of
   the 3 best-matching regions plus the world centroid (twice, as a
   stabiliser). Every photo is always answered; an unreadable photo falls
   back to the centroid, so no photo takes the 20,015 km unanswered penalty.

## Run

```
python3 -P -s main.py
```

with `MAKEFUN_ENV_URL` set to the trial env (the harness sets it). The bot
GETs `/photos`, GETs each photo, and POSTs its answers to `/answers`.

## Practice score

On the kit's 3 practice trials (75 photos): **545,509 km total**
(219,956 / 177,252 / 148,301 km per trial; median 7,069 km). Deterministic:
same photos give the same answers on every run.
