import cv2
import numpy as np
import os


def load_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Görüntü yüklenemedi: {image_path}")
    return image


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return rotated


def flip_horizontal(image):
    return cv2.flip(image, 1)


def flip_vertical(image):
    return cv2.flip(image, 0)


def translate_image(image, tx, ty):
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return translated


def scale_image(image, scale_factor):
    h, w = image.shape[:2]
    scaled = cv2.resize(
        image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_LINEAR
    )

    sh, sw = scaled.shape[:2]

    if scale_factor >= 1.0:
        start_x = (sw - w) // 2
        start_y = (sh - h) // 2
        scaled = scaled[start_y:start_y + h, start_x:start_x + w]
    else:
        canvas = np.zeros_like(image)
        start_x = (w - sw) // 2
        start_y = (h - sh) // 2
        canvas[start_y:start_y + sh, start_x:start_x + sw] = scaled
        scaled = canvas

    return scaled


def center_crop(image, crop_ratio=0.75):
    h, w = image.shape[:2]
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)

    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2

    cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]
    cropped_resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return cropped_resized


def shear_image(image, shear_factor=0.2):
    h, w = image.shape[:2]

    M = np.array([
        [1, shear_factor, 0],
        [0, 1, 0]
    ], dtype=np.float32)

    new_w = int(w + abs(shear_factor * h))

    sheared = cv2.warpAffine(
        image,
        M,
        (new_w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    sheared = cv2.resize(sheared, (w, h), interpolation=cv2.INTER_LINEAR)
    return sheared


def put_title(image, title):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 45), (255, 255, 255), -1)
    cv2.putText(
        output,
        title,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )
    return output


def resize_for_grid(image, size=(450, 300)):
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def make_grid(images, cols=2, cell_size=(450, 300), bg_color=(240, 240, 240)):
    labeled_images = [resize_for_grid(img, cell_size) for img in images]

    rows = int(np.ceil(len(labeled_images) / cols))
    cell_w, cell_h = cell_size

    grid = np.full(
        (rows * cell_h, cols * cell_w, 3),
        bg_color,
        dtype=np.uint8
    )

    for idx, img in enumerate(labeled_images):
        row = idx // cols
        col = idx % cols

        y1 = row * cell_h
        y2 = y1 + cell_h
        x1 = col * cell_w
        x2 = x1 + cell_w

        grid[y1:y2, x1:x2] = img

    return grid


def main():
    image_path = "/mnt/data/david-dibert-Huza8QOO3tc-unsplash.jpg"
    output_path = "geometric_transformations_collage.jpg"

    image = load_image(image_path)

    transformed_images = [
        put_title(image, "Original"),
        put_title(rotate_image(image, 25), "Rotation (+25 deg)"),
        put_title(flip_horizontal(image), "Horizontal Flip"),
        put_title(flip_vertical(image), "Vertical Flip"),
        put_title(translate_image(image, tx=120, ty=60), "Translation"),
        put_title(scale_image(image, 1.25), "Scaling (1.25x)"),
        put_title(center_crop(image, 0.75), "Center Crop"),
        put_title(shear_image(image, 0.25), "Shear")
    ]

    collage = make_grid(transformed_images, cols=2, cell_size=(500, 300))
    cv2.imwrite(output_path, collage)

    print(f"Birleşik görsel kaydedildi: {output_path}")


if __name__ == "__main__":
    main()
