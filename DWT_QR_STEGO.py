#DWT_QR_STEGO: Preston Mumphrey, Carlos Osuna, Luke Alvarado, Aradhnna Reddy
import sys
import subprocess
import argparse
import os
import secrets

import numpy as np
import cv2
import pywt
import qrcode
from PIL import Image
from pyzbar.pyzbar import decode

# -------------------------------------------------------------------------------------
# Required packages
REQUIRED_PACKAGES = [
    "numpy",
    "opencv-python",
    "pywt",
    "qrcode[pil]",
    "pillow",
    "pyzbar"
]

# -------------------------------------------------------------------------------------
# Install Python package via pip.
def install_package(package: str) -> None:
    print(f"[+] Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# -------------------------------------------------------------------------------------
# Makes sure REQUIRED_PACKAGES are installed before running the program.
def ensure_dependencies() -> None:
    import importlib
    for pkg in REQUIRED_PACKAGES:
        name = pkg.split('[')[0]
        try:
            importlib.import_module(name)
        except ImportError:
            install_package(pkg)

# -------------------------------------------------------------------------------------

def get_default_output_path(input_path: str, base_name: str) -> str:
    """
    Create a default output filename by replacing or appending an image extension.
    Args:
        input_path: Path to the cover or stego image file.
        base_name: Base name for the output file.
    Returns:
        A filename with .png or provided input extension.
        
    """
    _, ext = os.path.splitext(input_path)
    if ext.lower() not in ['.bmp', '.png']:
        ext = '.png'
    return f"{base_name}{ext}"

# -------------------------------------------------------------------------------------
def generate_qr_image(data: str) -> np.ndarray:
    """
    Create a high-error-correction QR code from text data provided.

    * Build QR with PIL, and then convert to grayscale.
    * Convert to numpy array and reduce bit depth by shifting to the right.

    Args:
        data: UTF-8 string.
    Returns:
        2D numpy array of QR intensity values on a range(0-127).
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4
    )
    qr.add_data(data)
    qr.make(fit=True)

    pil_img = qr.make_image(fill_color="black", back_color="white").convert('L')
    arr = np.array(pil_img)
    # Halve intensity values to soften embedding
    return np.right_shift(arr, 1)

# -------------------------------------------------------------------------------------
# Embed a QR code into a cover image using single-level DWT.
# Supports grayscale ('g') and color ('c') modes. (provided via -f)
def embed_message(
    cover_path: str,
    message_path: str,
    output_path: str,
    use_random: bool,
    mode: str
) -> None:
    """
    Embed text data as a QR code into a cover image.

    Steps:
    1. Load the cover image (g or c).
    2. Read text from file, or generate 128 bytes of random data.
    3. Generate a QR code array with generate_qr_image().
    4. Apply 2D DWT to cover (or the blue channel for color).
    5. Zero-init the LH detail subband, then add scaled QR pixels.
    6. Perform inverse DWT to reconstruct the stego image.
    7. End and save the stego image to the drive.

    Args:
        cover_path: Path to input cover image.
        message_path: Text file path.
        output_path: Filename for the stego output.
        use_random: If True, ignore message_path and embed random bytes.
        mode: 'g' for grayscale, 'c' for color embedding.
    """
    # 1. Load cover image
    if mode == 'g':
        cover = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    else:
        cover = cv2.imread(cover_path)
    if cover is None:
        print(f"[!] Failed to read image: {cover_path}")
        return

    # 2. Read or generate data
    if use_random:
        # Generate 128 random bytes, convert to hex string
        data_hex = secrets.token_bytes(128).hex()
    else:
        if not os.path.isfile(message_path):
            print(f"[!] File not found: {message_path}")
            return
        with open(message_path, 'r', encoding='utf-8') as f:
            data_hex = f.read()

    # 3. Create QR code array
    qr_arr = generate_qr_image(data_hex)
    alpha = 0.03  # Embedding strength coefficient

    if mode == 'g':
        # GRAYSCALE EMBEDDING
        cover = cover.astype(float)
        LL, (LH, HL, HH) = pywt.dwt2(cover, 'haar')

        # 4.Scale QR to LH dimentions
        
        qr_resized = cv2.resize(
            qr_arr, (LH.shape[1], LH.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        stego_subband = alpha * qr_resized

        # 5. Inverse DWT to reconstruct image
        stego = pywt.idwt2((LL, (stego_subband, HL, HH)), 'haar')
        stego_img = np.clip(stego, 0, 255).astype(np.uint8)

    else:
        # COLOR EMBEDDING (Blue Chanel)
        B, G, R = cv2.split(cover)
        B = B.astype(float)
        LL, (LH, HL, HH) = pywt.dwt2(B, 'haar')

        qr_resized = cv2.resize(
            qr_arr, (LH.shape[1], LH.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        stego_subband = alpha * qr_resized

        # Reconstruct blue channel and reassemble color image
        stego_B = pywt.idwt2((LL, (stego_subband, HL, HH)), 'haar')
        stego_B = np.clip(stego_B, 0, 255).astype(np.uint8)
        # stego_img cv2.merge((stego_B, G, R))
        stego_B_resized = cv2.resize(stego_B, (G.shape[1], G.shape[0]), interpolation=cv2.INTER_NEAREST)
        stego_img = cv2.merge((stego_B_resized, G, R))

    # 6. Save stego image to drive
    cv2.imwrite(output_path, stego_img)
    print(f"[+] Stego image saved to: {output_path}")

# -------------------------------------------------------------------------------------
# Extract and decode a QR code from a stego-image.
def extract_message(
    stego_path: str,
    mode: str,
    output_path: str = None
) -> None:
    """
    Recover embedded QR message from a stego image.

    Steps:
    1. Load stego image (g or c).
    2. For color, split and use blue channel.
    3. Apply 2D Haar DWT and isolate LH band.
    4. Left-shift subband to approximate QR bits.
    5. Threshold to binary image, resize to QR size.
    6. Save and decode.

    Args:
        stego_path: Path to the stego image file.
        mode: 'g' or 'c' for extraction mode.
        output_path: Optional file to write decoded text.
    """
    # 1. Load and prepare image
    if mode == 'g':
        img = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE).astype(float)
    else:
        img = cv2.imread(stego_path)
        if img is not None:
            B, _, _ = cv2.split(img)
            img = B.astype(float)
    if img is None:
        print(f"[!] Failed to load image: {stego_path}")
        return

    # 2. DWT decomposition
    LL, (LH, HL, HH) = pywt.dwt2(img, 'haar')

    # 3. Approximate original QR bits
    shifted = np.left_shift(LH.astype(np.uint8), 1)
    binary = np.where(
        shifted == np.min(shifted),
        0,
        255
    ).astype(np.uint8)

    # 4. Resize to the subband dimensions and save
    qr_img = cv2.resize(
        binary, (LH.shape[1], LH.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    pil_qr = Image.fromarray(qr_img)
    pil_qr.save("extracted_qr.png")
    print("[+] Extracted QR saved as 'extracted_qr.png'")

    # 5. Decode
    decoded = decode(pil_qr)
    if decoded:
        message = decoded[0].data.decode('utf-8')
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(message)
            print(f"[+] Message saved to: {output_path}")
        else:
            print(f"[+] Decoded message: {message}")
    else:
        print("[-] No QR code detected.")

# -------------------------------------------------------------------------------------
# Parse CLI arguments and invoke hide or extract routines.
def main() -> None:
    # Ensure we have all required dependencies
    ensure_dependencies()

    parser = argparse.ArgumentParser(
        description="DWT QR Steganography Tool"
    )
    subp = parser.add_subparsers(
        dest='command',
        required=True
    )

    # 'hide' command
    hide = subp.add_parser(
        'hide', help='Embed a message as QR in an image'
    )
    hide.add_argument(
        '-m', '--message', required=True,
        help="Path to text file, or 'random' to embed random bytes."
    )
    hide.add_argument(
        '-c', '--cover', required=True,
        help='Cover image file path.'
    )
    hide.add_argument(
        '-f', '--format', choices=['g', 'c'], default='c',
        help="Embedding mode: 'g'=grayscale, 'c'=color."
    )
    hide.add_argument(
        '-o', '--output', default=None,
        help='Destination filename for stego image.'
    )

    # 'extract' command
    extract = subp.add_parser(
        'extract', help='Extract embedded QR message'
    )
    extract.add_argument(
        '-s', '--stego', required=True,
        help='Input stego image file.'
    )
    extract.add_argument(
        '-f', '--format', choices=['g', 'c'], default='c',
        help="Extraction mode: 'g'=grayscale, 'c'=color."
    )
    extract.add_argument(
        '-o', '--output', default=None,
        help='Optional path to save decoded text.'
    )

    args = parser.parse_args()

    if args.command == 'hide':
        # Determine output filename if not provided
        output = (
            args.output or
            get_default_output_path(args.cover, 'stego_output')
        )
        # Invoke embedding routine
        embed_message(
            cover_path=args.cover,
            message_path=args.message,
            output_path=output,
            use_random=(args.message.lower() == 'random'),
            mode=args.format
        )
    else:
        # Invoke extraction routine
        extract_message(
            stego_path=args.stego,
            mode=args.format,
            output_path=args.output
        )

# -------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
