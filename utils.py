import base64


def decode_image(imagestring, filename):
    img = base64.b64decode(imagestring)

    with open(filename, 'wb') as f:
        f.write(img)
        f.close()