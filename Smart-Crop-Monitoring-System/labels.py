"""Small helpers that turn model class names into readable text.

Class names look like  Tomato___Early_blight  (crop ___ condition).
These helpers only change how the name is DISPLAYED. They never change the
class names stored in class_names.json, so model compatibility is untouched.
"""

SEPARATOR = "___"


def crop_name(label):
    """'Pepper,_bell___Bacterial_spot' -> 'Pepper, bell'"""
    crop, _, _ = label.partition(SEPARATOR)
    return crop.replace("_", " ").strip()


def condition_name(label):
    """'Tomato___Early_blight' -> 'Early blight'; '...___healthy' -> 'Healthy'"""
    _, _, condition = label.partition(SEPARATOR)
    condition = condition.replace("_", " ").strip()
    if condition.lower() == "healthy":
        return "Healthy"
    return condition or label.replace("_", " ")


def is_healthy(label):
    return label.lower().endswith(SEPARATOR + "healthy")
