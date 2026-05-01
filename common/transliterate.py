IASTToSlpDict = {
	chr(0x1E43): "M", # ṃ
    chr(0x1E41): "M", # ṁ
	chr(0x1E25): "H",
	chr(0x101):  "A",
	chr(0x12B):  "I",
	chr(0x16B):  "U",
	chr(0x1E5B): "f", # ṛ
    chr(0x1E5D): "F", # ṝ
	chr(0x1E37): "x",
	chr(0x1E45): "N",
	chr(0xF1):   "Y",
	chr(0x1E6D): "w",
	chr(0x1E0D): "q",
	chr(0x1E47): "R",
	chr(0x15B):  "S",
	chr(0x1E63): "z",
}

IASTToSlpString = {
	"ai":               "E",
	"au":               "O",
	"kh":               "K",
	"gh":               "G",
	"ch":               "C",
	"jh":               "J",
	chr(0x1E6D) + "h":    "W",
	chr(0x1E0D) + "h":    "Q",
	"th":               "T",
	"dh":               "D",
	"ph":               "P",
	"bh":               "B",
}

SlpToIAST = {
	"M": chr(0x1E43), # ṃ
	"H": chr(0x1E25),
	"A": chr(0x101),
	"I": chr(0x12B),
	"U": chr(0x16B),
	"f": chr(0x1E5B),
    "F": chr(0x1E5D),
	"x": chr(0x1E37),
	"N": chr(0x1E45),
	"Y": chr(0xF1),
	"w": chr(0x1E6D),
	"q": chr(0x1E0D),
	"R": chr(0x1E47),
	"S": chr(0x15B),
	"z": chr(0x1E63),
	"E": "ai",
	"O": "au",
	"K": "kh",
	"G": "gh",
	"C": "ch",
	"J": "jh",
	"W": chr(0x1E6D) + "h",
	"Q": chr(0x1E0D) + "h",
	"T": "th",
	"D": "dh",
	"P": "ph",
	"B": "bh",
}

def slpToIAST(text):
    for src, trg in SlpToIAST.items():
        text = text.replace(src, trg)
    return text


def IASTToSlp(text):
    for src, trg in IASTToSlpString.items():
        text = text.replace(src, trg)
    for src, trg in IASTToSlpDict.items():
        text = text.replace(src, trg)
    return text
