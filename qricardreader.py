#!/usr/bin/env python3
"""
ICard QR Reader & Verifier
Reads the signed QR from icard_back.png, verifies the RSA-SHA256
signature against icard_sign.cer, and prints all card details
including embedded photo and full certificate/signature info.

Usage:
  python3 qricardreader.py [--image icard/icard_back.png] [--cert icard_sign.cer]

Outputs:
  icard_photo.jp2   - embedded face (raw J2K codestream)
  icard_photo.png   - converted face photo
"""

import sys
import argparse
import gzip
import hashlib
import datetime
from pathlib import Path

sys.set_int_max_str_digits(0)

import zxingcpp
import cv2
import numpy as np
from PIL import Image
from cryptography.x509 import load_pem_x509_certificate
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from cryptography.x509.oid import NameOID


# ─────────────────────────── field map ───────────────────────────────

# Mirrors the field order in make_icard.py → build_qr_payload()
FIELD_NAMES = [
    "Reference ID",    # 0: last4ID + YYYYMMDDHHMMSSmmm  (not printed as-is)
    "Name",            # 1
    "DOB",             # 2
    "Gender",          # 3
    "Care Of",         # 4
    "District",        # 5
    "Landmark",        # 6
    "House",           # 7
    "Location",        # 8
    "Pincode",         # 9
    "Post Office",     # 10
    "State",           # 11
    "VTC",             # 12
    "Sub District",    # 13
    "PO",              # 14
    "Mobile Last 4",   # 15
]

JP2_SOC  = b'\xff\x4f\xff\x51'   # JPEG2000 Start Of Codestream
JP2_EOC  = b'\xff\xd9'           # JPEG2000 End Of Codestream


# ─────────────────────────── helpers ─────────────────────────────────

def section(title: str):
    width = 55
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def get_name_attr(name, oid, default="N/A"):
    try:
        return name.get_attributes_for_oid(oid)[0].value
    except (IndexError, Exception):
        return default


def decode_qr(image_path: str) -> str:
    img = Image.open(image_path)
    results = zxingcpp.read_barcodes(img)
    if not results:
        raise RuntimeError(f"No QR code detected in: {image_path}")
    r = results[0]
    print(f"QR format   : {r.format}")
    print(f"Content type: {r.content_type}")
    print(f"Digits      : {len(r.text)}")
    return r.text


def decompress_payload(decimal_str: str) -> bytes:
    big_int   = int(decimal_str)
    num_bytes = (big_int.bit_length() + 7) // 8
    raw       = big_int.to_bytes(num_bytes, byteorder='big')
    if raw[:2] != b'\x1f\x8b':
        raise RuntimeError("Payload is not gzip-compressed — unexpected format")
    return gzip.decompress(raw)


def split_payload(decompressed: bytes, sig_len: int):
    """Split decompressed bytes into signed_data, signature, masked_email."""
    # signature is the last sig_len bytes of the (pre-email) block.
    # masked email is everything after the signature.
    # We find the boundary by locating the JP2 EOC within the data
    # and then taking sig_len bytes after that block ends.

    # Find last JP2 EOC to locate end of photo
    eoc_pos = decompressed.rfind(JP2_EOC)
    if eoc_pos == -1:
        raise RuntimeError("JP2 EOC marker not found in payload")

    photo_end    = eoc_pos + 2          # byte after EOC
    sig_start    = photo_end
    sig_end      = sig_start + sig_len
    signed_data  = decompressed[:sig_start]
    signature    = decompressed[sig_start:sig_end]
    masked_email = decompressed[sig_end:].decode('utf-8', errors='replace')

    return signed_data, signature, masked_email


def verify_signature(signed_data: bytes, signature: bytes, public_key) -> tuple[bool, str]:
    """Returns (is_valid, message)."""
    try:
        public_key.verify(signature, signed_data, padding.PKCS1v15(), hashes.SHA256())
        return True, "SIGNATURE VALID ✓"
    except Exception as e:
        return False, f"SIGNATURE INVALID ✗  ({e!r})"


def extract_fields(signed_data: bytes):
    """
    Parse the 0xFF-delimited field blob out of signed_data.
    Layout: V5 \xff <email_flag> \xff <fields...> \xff <JP2 bytes>
    Returns: version, email_flag, fields_list, photo_jp2_bytes
    """
    parts   = signed_data.split(b'\xff')
    version = parts[0].decode(errors='replace')
    email_flag = parts[1].decode(errors='replace')

    remaining  = b'\xff'.join(parts[2:])

    jp2_start = remaining.find(JP2_SOC)
    if jp2_start == -1:
        raise RuntimeError("JP2 SOC marker not found — photo not embedded")

    field_blob = remaining[:jp2_start]
    photo_jp2  = remaining[jp2_start:]          # includes trailing EOC

    fields = [f.decode('utf-8', errors='replace') for f in field_blob.split(b'\xff')]
    return version, email_flag, fields, photo_jp2


def parse_ref_id(ref_id: str):
    """Split 'LLLLYYYYMMDDHHMMSS000' into last4 and a datetime."""
    last4 = ref_id[:4]
    ts    = ref_id[4:]
    try:
        dt = datetime.datetime.strptime(ts[:14], "%Y%m%d%H%M%S")
        signed_at = dt.strftime("%d %b %Y  %H:%M:%S")
    except ValueError:
        signed_at = ts
    return last4, signed_at


def save_photo(photo_jp2: bytes, out_dir: Path) -> bool:
    jp2_path = out_dir / "icard_photo.jp2"
    png_path = out_dir / "icard_photo.png"

    # Strip anything after EOC before saving
    eoc = photo_jp2.find(JP2_EOC)
    clean = photo_jp2[:eoc + 2] if eoc != -1 else photo_jp2
    jp2_path.write_bytes(clean)

    arr = cv2.imdecode(np.frombuffer(clean, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        print(f"  Saved raw codestream : {jp2_path}")
        print("  PNG conversion failed (try: convert icard_photo.jp2 icard_photo.png)")
        return False
    cv2.imwrite(str(png_path), arr)
    print(f"  Saved JP2 : {jp2_path}  ({len(clean)} bytes)")
    print(f"  Saved PNG : {png_path}  ({arr.shape[1]}×{arr.shape[0]} px)")
    return True


def print_cert_details(cert, signature: bytes, signed_data: bytes):
    section("CERTIFICATE DETAILS")
    subj = cert.subject
    issr = cert.issuer

    print(f"  Subject")
    print(f"    Common Name : {get_name_attr(subj, NameOID.COMMON_NAME)}")
    print(f"    Organisation: {get_name_attr(subj, NameOID.ORGANIZATION_NAME)}")
    print(f"    Org Unit    : {get_name_attr(subj, NameOID.ORGANIZATIONAL_UNIT_NAME)}")
    print(f"    State       : {get_name_attr(subj, NameOID.STATE_OR_PROVINCE_NAME)}")
    print(f"    Country     : {get_name_attr(subj, NameOID.COUNTRY_NAME)}")

    print(f"\n  Issuer")
    print(f"    Common Name : {get_name_attr(issr, NameOID.COMMON_NAME)}")
    print(f"    Organisation: {get_name_attr(issr, NameOID.ORGANIZATION_NAME)}")

    print(f"\n  Validity")
    print(f"    Not Before  : {cert.not_valid_before_utc.strftime('%d %b %Y  %H:%M:%S UTC')}")
    print(f"    Not After   : {cert.not_valid_after_utc.strftime('%d %b %Y  %H:%M:%S UTC')}")

    now = datetime.datetime.now(datetime.timezone.utc)
    expired = now > cert.not_valid_after_utc
    not_yet = now < cert.not_valid_before_utc
    if expired:
        print(f"    Status      : ⚠  EXPIRED")
    elif not_yet:
        print(f"    Status      : ⚠  NOT YET VALID")
    else:
        print(f"    Status      : ✓  Currently valid")

    pub = cert.public_key()
    if isinstance(pub, RSAPublicKey):
        print(f"\n  Public Key")
        print(f"    Algorithm   : RSA-{pub.key_size}")
        print(f"    Exponent    : {pub.public_numbers().e}")

    print(f"\n  Serial No.  : {hex(cert.serial_number)}")
    fp = cert.fingerprint(hashes.SHA256()).hex()
    print(f"  SHA-256 FP  : {':'.join(fp[i:i+2] for i in range(0, len(fp), 2))}")

    section("SIGNATURE DETAILS")
    print(f"  Algorithm   : RSA-{pub.key_size if isinstance(pub, RSAPublicKey) else '?'} / SHA-256 / PKCS1v15")
    print(f"  Length      : {len(signature)} bytes")
    print(f"  Hex (first 32 bytes) : {signature[:32].hex()}")

    # Compute SHA-256 of the signed region to show what was protected
    digest = hashlib.sha256(signed_data).hexdigest()
    print(f"\n  SHA-256 of signed region:")
    print(f"    {digest[:32]}")
    print(f"    {digest[32:]}")
    print(f"\n  Signed region covers:")
    print(f"    • All demographic fields (name, DOB, gender, address)")
    print(f"    • Embedded JP2 face photo bytes")
    print(f"    • Card ID last-4 + generation timestamp")
    print(f"    (Masked email is NOT signed — same as UIDAI design)")


# ─────────────────────────── main ────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ICard QR Reader & Verifier")
    parser.add_argument("--image", default="icard/icard_back.png",
                        help="Path to the card back image")
    parser.add_argument("--cert",  default="icard_sign.cer",
                        help="PEM certificate from gen_cert.py")
    parser.add_argument("--out",   default=".",
                        help="Directory to save extracted photo")
    args = parser.parse_args()

    image_path = Path(args.image)
    cert_path  = Path(args.cert)
    out_dir    = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_path.is_file():
        print(f"Error: image not found: {image_path}"); sys.exit(1)
    if not cert_path.is_file():
        print(f"Error: certificate not found: {cert_path}"); sys.exit(1)

    # --- Load certificate ---
    cert       = load_pem_x509_certificate(cert_path.read_bytes())
    public_key = cert.public_key()
    sig_len    = public_key.key_size // 8     # 256 for RSA-2048

    section("QR DECODE")
    decimal_str  = decode_qr(str(image_path))

    section("PAYLOAD")
    decompressed = decompress_payload(decimal_str)
    print(f"  Decompressed size : {len(decompressed)} bytes")

    signed_data, signature, masked_email = split_payload(decompressed, sig_len)
    print(f"  Signed region     : {len(signed_data)} bytes")
    print(f"  Signature         : {len(signature)} bytes")

    section("SIGNATURE VERIFICATION")
    is_valid, msg = verify_signature(signed_data, signature, public_key)
    print(f"\n  {msg}\n")
    if not is_valid:
        print("  ⚠  Details below are from an UNVERIFIED payload.")

    section("CARD DETAILS")
    version, email_flag, fields, photo_jp2 = extract_fields(signed_data)
    print(f"  QR Version        : {version}")

    flag_map = {"0": "None", "1": "Email only", "2": "Mobile only", "3": "Email + Mobile"}
    print(f"  Contact flag      : {email_flag}  ({flag_map.get(email_flag, 'Unknown')})")

    ref_id = fields[0] if fields else ""
    last4_id, signed_at = parse_ref_id(ref_id)
    print(f"  Card ID last 4    : {last4_id}")
    print(f"  QR signed at      : {signed_at}")
    print()

    for i, val in enumerate(fields):
        if i >= len(FIELD_NAMES):
            break
        label = FIELD_NAMES[i]
        if label == "Reference ID":
            continue
        if val:
            print(f"  {label:<16}: {val}")

    # Mobile last 4 — strip the XXXXXX mask
    if len(fields) > 15 and fields[15]:
        digits = ''.join(c for c in fields[15] if c.isdigit())
        print(f"  {'Mobile last 4':<16}: {digits}")

    if masked_email:
        print(f"  {'Masked Email':<16}: {masked_email}")

    section("EMBEDDED PHOTO")
    save_photo(photo_jp2, out_dir)

    print_cert_details(cert, signature, signed_data)

    section("SUMMARY")
    status = "✓  VALID — card details are authentic and untampered" if is_valid \
             else "✗  INVALID — card may be forged or tampered"
    print(f"\n  {status}\n")


if __name__ == "__main__":
    main()
