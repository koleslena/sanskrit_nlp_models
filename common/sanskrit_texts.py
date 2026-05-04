import os
from enum import Enum
from dotenv import load_dotenv 

load_dotenv()

TEXTS_DIR = os.getenv("SANSKRIT_TEXTS_DIR")

class Datasources(str, Enum):
    # Эпический санскрит и Пураны 
    MAHABHARATA = "Mahabharata"
    RAMAYANA = "Ramayana"
    HARIVAMSHA = "Harivamsha"

    # Purāṇas:
    VISHNUPURANA = "Vishnupurana"
    BHAGAVATAPURANA = "Bhagavatapurana"
    SHIVAPURANA = "Shivapurana"
    AGNIPURANA = "Agnipurana" 
    GARUDAPURANA = "Garudapurana"
    KALIKAPURANA = "Kalikapurana"
    KURMAPURANA = "Kurmapurana"
    LINGAPURANA = "Lingapurana"
    MATSYAPURANA = "Matsyapurana"
    NARASIMHAPURANA = "Narasimhapurana"
    SKANDAPURANA = "Skandapurana"
    VARAHAPURANA = "Varahapurana"

    # Дидактическая проза (Сказки и басни)
    HITOPADESHA = 'Hitopadesha'
    PANCATANTRA = "Tantrakhyayika"
    VETALAPANCAVIMSHATIKA = "Vetalapancavimshatika"
    SHUKASAPTATI = "Shukasaptati"
    KATHASARITASAGARA = "Kathasaritsagara"

    # Аюрведа и естественные науки (Shāstra)
    # Основы: 
    CARAKA = "Carakasamhita"
    SUSHRUTA = "Sushrutasamhita"
    ASHTANGAHRDAYA = "Ashtangahrdayasamhita"
    ASHTANGASAMGRAHA = "Ashtangasamgraha"
    # Нигханту (словари трав): 
    ASHTANGANIGHANTU = "Ashtanganighantu"
    BIJANIGHANTU = "Bijanighantu"
    DHANVANTARI = "Dhanvantarinighantu"
    KAIYADEVA = "Kaiyadevanighantu"
    MADANAPALA = "Madanapalanighantu"
    RAJA = "Rajanighantu"
    # Алхимия и Расаяна: 
    RASAHRDAYA = "Rasahrdayatantra"
    RASARATNA = "Rasaratnasamuccaya"
    RASARNAVA = "Rasarnava"
    RASENDRA = "Rasendracintamani"
    # Другое: 
    ARTHASHASTRA = "Arthashastra" # политика
    KRSHIPARASHARA = "Krshiparashara" # агро
    MANUSMRTI = "Manusmrti" # законы

    # Философия, Йога и Тантра
    # Йога/Хатха-йога:
    GHERANDA = "Gherandasamhita"
    GORAKSHA = "Gorakshashataka"
    HATHAYOGAPRADIPIKA = "Hathayogapradipika"
    YOGASUTRA = "Yogasutra"

    # Тантра/Кашмирский Шиваизм:
    TANTRALOKA = "Tantraloka"
    SHIVASUTRA = "Shivasutra"
    MRGENDRATANTRA = "Mrgendratantra"
    SPANDAKARIKA = "Spandakarika"

    # Философия: 
    SAMKHYAKARIKA = "Samkhyakarika"
    NYAYASUTRA = "Nyayasutra"
    VAISHESHIKASUTRA = "Vaisheshikasutra"

    AMARAKOSHA = "Amarakosha"

    @classmethod
    def get_datasource_list(cls):
        return [source.value for source in cls]