from PIL import Image
from math import atan, degrees




if __name__ == '__main__':
    with Image.open('IMG_20230423_222441.jpg') as img:
        exif_data = img._getexif()
        focal_length = exif_data[37386]
        sensor_size = (exif_data[37377], exif_data[37378])
        fov = degrees(2 * atan(sensor_size[1] / (2 * focal_length)))

    print(fov)