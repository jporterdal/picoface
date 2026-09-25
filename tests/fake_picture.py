"""A minimal stand-in for mediaComp's `Picture`, for tests that don't need mediaComp.

It offers only what picoface relies on (a constructor that takes a Pillow
image, and `getImage()`), plus mediaComp-style pixel access in RGB triples so
tests can draw and inspect the way students do.
"""

from PIL import Image, ImageDraw


class FakePicture:
    def __init__(self, image: Image.Image):
        self.image = image

    @classmethod
    def blank(cls, width: int, height: int, color=(255, 255, 255)) -> "FakePicture":
        return cls(Image.new("RGB", (width, height), color))

    def getImage(self) -> Image.Image:
        return self.image

    def getWidth(self) -> int:
        return self.image.width

    def getHeight(self) -> int:
        return self.image.height

    def getBasicPixel(self, x: int, y: int) -> tuple[int, int, int]:
        """The pixel at (x, y) as an (r, g, b) triple; fails, as mediaComp does,
        if the image is not in an RGB mode."""
        red, green, blue = self.image.getpixel((x, y))
        return red, green, blue

    def setBasicPixel(self, x: int, y: int, rgb) -> None:
        self.image.putpixel((x, y), tuple(rgb))

    def addOvalFilled(self, color, x: int, y: int, w: int, h: int) -> None:
        ImageDraw.Draw(self.image).ellipse([x, y, x + w, y + h], fill=tuple(color))
