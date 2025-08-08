# DWT_QR_STEGO

DWT_QR_STEGO is a steganographic program that hides text as a qr image into a chosen cover and allows extraction of the same image to reveal the text.

## Installation

Upon running the executable for the first time, the necessary python packages will be installed, and upon every subsequent execute the dependencies will be checked.

## Usage


Embedding:
```python
python DWT_QR_STEGO.py hide -m mySecret.txt -c myCover.png -f g -o stegoImg.png
returns stegoImg.png
```
Flags:

-m is the message flag and can be and can be provided with text file. If other data is to be hidden, it must be converted to txt which essentially means reading the file as binary into a txt file. This flag must be provided with input. The word random can be provided which then embeds 128 bytes of random data into the cover image

-c is the cover image flag and can be provided with any type of png or bmp image. As jpgs are already lossy and compressed, the DWT transform does not perform as expected on this file format. This flag must be provided with input.

-f is the format flag and can only be provided with the values of g for grayscale mode and c for color mode. This flag is not mandatory as color is the default mode.

-o is the output path flag. This must be a png or bmp but does not depend on the file type of the cover, as png can be saved as a bmp and vice versa while preserving functionality. This flag is not mandatory, as the default output path is stego_output followed by the file extension matching the cover image.

Extracting:
```python
python DWT_QR_STEGO.py extract -s stegoImg.png -f c -o message.txt
returns message.txt and extracted_qr.png
```
Flags:

-s is the stego-image flag and must be provided with an image generated from this program to ensure full functionality. This will be either a png or bmp

-f is the format flag and can only be provided with the values of g for grayscale mode and c for color mode. This flag is not mandatory as color is the default mode, however this flag must be the same as the chosen option provided when embedding.

-o is the output path flag. This can be as a txt file, but if the data hidden was of a different format that was then converted to text, the file extension of that original format can be provided. This flag is not mandatory, as the default action is to print the hidden text to the terminal.

## Notes
Capacity of the highest version of QR codes being version 40 is 2953 bytes.
If a file is larger than this, it must be split into different stego-images. The best usage for this program is embedding a web address, which is what QR codes are useful for.

Robustness of the program should prove acceptable, as QR codes are designed by nature to be readable even when stretched, contorted, and compressed, and is also the primary benefit of this steganographic method over others.

## Credits
Preston Mumphrey, Carlos Osuna, Luke Alvarado

Priya, K., et al. “DWT based QR Steganography.” Journal of Physics: Conference Series, vol. 1917, no. 1, 1 June 2021, p. 012020, https://doi.org/10.1088/1742-6596/1917/1/012020.


