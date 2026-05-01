from common.transliterate import slpToIAST, IASTToSlp

slp_to_iast_cases = [{ 
			'src': "acalaSrezWa",
			'trg': "acalaśreṣṭha",
		}, {
			'src':    "ajo'pi sannavyayAtmA BUtAnAmISvaro'pi san |",
			'trg': "ajo'pi sannavyayātmā bhūtānāmīśvaro'pi san |",
		}, {
			'src':    "aByutTAnamaDarmasya tadA''tmAnaM sfjAmyaham ||4.7||",
			'trg': "abhyutthānamadharmasya tadā''tmānaṃ sṛjāmyaham ||4.7||",
		}, {
			'src':    "sa evAyaM mayA te'dya yogaH proktaH purAtanaH |",
			'trg': "sa evāyaṃ mayā te'dya yogaḥ proktaḥ purātanaḥ |",
		}, {
			'src':    "bahUni me vyatItAni janmAni tava cArjuna|",
			'trg': "bahūni me vyatītāni janmāni tava cārjuna|",
		}, {
			'src':    "prakftiM svAmaDizWAya saMBavAmyAtmamAyayA ||4.6||",
			'trg': "prakṛtiṃ svāmadhiṣṭhāya saṃbhavāmyātmamāyayā ||4.6||",
		}]
iast_to_slp_cases = [{ 
			'src':    "acalaśreṣṭha",
			'trg': "acalaSrezWa",
		}, {
			'src':    "ajo'pi sannavyayātmā bhūtānāmīśvaro'pi san |",
			'trg': "ajo'pi sannavyayAtmA BUtAnAmISvaro'pi san |",
		}, {
			'src':    "abhyutthānamadharmasya tadā''tmānaṃ sṛjāmyaham ||4.7||",
			'trg': "aByutTAnamaDarmasya tadA''tmAnaM sfjAmyaham ||4.7||",
		}, {
			'src':    "sa evāyaṃ mayā te'dya yogaḥ proktaḥ purātanaḥ |",
			'trg': "sa evAyaM mayA te'dya yogaH proktaH purAtanaH |",
		}, {
			'src':    "bahūni me vyatītāni janmāni tava cārjuna|",
			'trg': "bahUni me vyatItAni janmAni tava cArjuna|",
		}, {
			'src':    "prakṛtiṃ svāmadhiṣṭhāya saṃbhavāmyātmamāyayā ||4.6||",
			'trg': "prakftiM svAmaDizWAya saMBavAmyAtmamAyayA ||4.6||",
		}]

def test_slpToIAST():
    for case in slp_to_iast_cases:
        assert slpToIAST(case['src']) == case['trg']

def test_IASTToSlp():
    for case in iast_to_slp_cases:
        assert IASTToSlp(case['src']) == case['trg']