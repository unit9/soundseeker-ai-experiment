from PIL import ImageStat, Image
from io import BytesIO

#move to settings
IMAGE_DEST_SIZE = (600, 600)

def load_image(path: str, resize=True) -> Image:
    """
    " Loads and resizes the image
    """
    im = Image.open(path).convert("RGB")
    im.thumbnail(IMAGE_DEST_SIZE, Image.ANTIALIAS)
    return im


def pil_to_byteIO(img: Image) -> BytesIO:
    """
    " Converts to Pil to BytesIO nescecery for google vision api
    """
    ret = BytesIO()
    img.save(ret, "JPEG", quality=95)
    return ret.getvalue()


def get_image_stats(image: Image):
    greyscale = image.convert('L')
    stat = ImageStat.Stat(greyscale)

    stat_dict = {}
    for i, field in enumerate(('mean', 'median', 'rms', 'var', 'stddev', 'extrema')):

        v = getattr(stat, field)[0]
        if isinstance(v, tuple):
            stat_dict[field + '_min'] = v[0]
            stat_dict[field + '_max'] = v[1]
        else:
            stat_dict[field] = v

    return stat_dict