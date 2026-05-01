import os
from enum import Enum

TEXTS_DIR = os.environ.get("SANSKRIT_TEXTS_DIR")

class Datasources(str, Enum):
    # Эпический санскрит и Пураны 
    MAHABHARATA = "Mahābhārata"
    RAMAYANA = "Rāmāyaṇa"
    HARIVAMSHA = "Harivaṃśa"

    # Purāṇas:
    VISHNUPURANA = "Viṣṇupurāṇa"
    BHAGAVATAPURANA = "Bhāgavatapurāṇa"
    SHIVAPURANA = "Śivapurāṇa"
    AGNIPURANA = "Agnipurāṇa" 
    GARUDAPURANA = "Garuḍapurāṇa"
    KALIKAPURANA = "Kālikāpurāṇa"
    KURMAPURANA = "Kūrmapurāṇa"
    LINGAPURANA = "Liṅgapurāṇa"
    MATSYAPURANA = "Matsyapurāṇa"
    NARASIMHAPURANA = "Narasiṃhapurāṇa"
    SKANDAPURANA = "Skandapurāṇa"
    VARAHAPURANA = "Varāhapurāṇa"

    # Дидактическая проза (Сказки и басни)
    HITOPADESHA = 'Hitopadeśa'
    PANCATANTRA = "Tantrākhyāyikā"
    VETALAPANCAVIMSHATIKA = "Vetālapañcaviṃśatikā"
    SHUKASAPTATI = "Śukasaptati"
    KATHASARITASAGARA = "Kathāsaritsāgara"

    # Аюрведа и естественные науки (Śāstra)
    # Основы: 
    CARAKA = "Carakasaṃhitā"
    SUSHRUTA = "Suśrutasaṃhitā"
    ASHTANGAHRDAYA = "Aṣṭāṅgahṛdayasaṃhitā"
    ASHTANGASAMGRAHA = "Aṣṭāṅgasaṃgraha"
    # Нигханту (словари трав): 
    ASHTANGANIGHANTU = "Aṣṭāṅganighaṇṭu"
    BIJANIGHANTU = "Bījanighaṇṭu"
    DHANVANTARI = "Dhanvantarinighaṇṭu"
    KAIYADEVA = "Kaiyadevanighaṇṭu"
    MADANAPALA = "Madanapālanighaṇṭu"
    RAJA = "Rājanighaṇṭu"
    # Алхимия и Расаяна: 
    RASAHRDAYA = "Rasahṛdayatantra"
    RASARATNA = "Rasaratnasamuccaya"
    RASARNAVA = "Rasārṇava"
    RASENDRA = "Rasendracintāmaṇi"
    # Другое: 
    ARTHASHASTRA = "Arthaśāstra" # политика
    KRSHIPARASHARA = "Kṛṣiparāśara" # агро
    MANUSMRTI = "Manusmṛti" # законы

    # Философия, Йога и Тантра
    # Йога/Хатха-йога:
    GHERANDA = "Gheraṇḍasaṃhitā"
    GORAKSHA = "Gorakṣaśataka"
    HATHAYOGAPRADIPIKA = "Haṭhayogapradīpikā"
    YOGASUTRA = "Yogasūtra"

    # Тантра/Кашмирский Шиваизм:
    TANTRALOKA = "Tantrāloka"
    SHIVASUTRA = "Śivasūtra"
    MRGENDRATANTRA = "Mṛgendratantra"
    SPANDAKARIKA = "Spandakārikā"

    # Философия: 
    SAMKHYAKARIKA = "Sāṃkhyakārikā"
    NYAYASUTRA = "Nyāyasūtra"
    VAISHESHIKASUTRA = "Vaiśeṣikasūtra"

    AMARAKOSHA = "Amarakośa"

    def get_datasource_list():
        return [source.value for source in Datasources]