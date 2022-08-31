from PIL import Image


def resize_image_if_needed(image_path, max_width=4096, max_height=3200):
    with Image.open(image_path) as img:
        width, height = img.size

        # Check if the image exceeds the maximum dimensions
        if width > max_width or height > max_height:
            # Calculate the new size maintaining the aspect ratio
            ratio = min(max_width/width, max_height/height)
            new_size = (int(width * ratio), int(height * ratio))
            resized_img = img.resize(new_size, Image.LANCZOS)

            # Save the resized image
            resized_img.save(image_path)
            return f"Image resized to {new_size[0]}x{new_size[1]}px"
        else:
            return "No resizing needed, image within limits."