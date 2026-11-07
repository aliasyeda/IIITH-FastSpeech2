import re
def read_phones(path):
    txt = open(path, encoding="utf-8").read()
    for t in re.split(r"item \[\d+\]:", txt)[1:]:
        m = re.search(r'name = "(.*?)"', t)
        if m and m.group(1).lower() == "phones":
            return [(float(a), float(b), c) for a, b, c in
                    re.findall(r'xmin = ([\d.eE+-]+)\s*xmax = ([\d.eE+-]+)\s*text = "(.*?)"', t, flags=re.S)]
    raise ValueError(f"no phones tier in {path}")
