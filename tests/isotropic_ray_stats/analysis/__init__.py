stat_codes = ("An", "Gn", "W", "K")

def _validate_code(code):
    if not isinstance(code, str):
        raise TypeError("Statistic code must be a string identifier")
    if not code in stat_codes:
        raise ValueError("Statistic code %s does not exist" % code)
