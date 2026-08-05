"""
Download international stock data via yfinance.
Covers Japan, Europe (UK/DE/FR/CH), Hong Kong, and EM (Brazil, India, Korea).
Saves adjusted close prices to parquet.
"""

import yfinance as yf
import pandas as pd
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

START = "2000-01-01"
END = "2024-12-31"
BATCH_SIZE = 5
SLEEP_BETWEEN_BATCHES = 3.0  # seconds
OUTPUT = "D:/code/data/intl_price.parquet"

# ── Ticker lists ──────────────────────────────────────────────────────────────

JAPAN = [
    # Automotive & Parts
    "7203.T", "7267.T", "7201.T", "6902.T", "7259.T", "7269.T", "7270.T",
    "5334.T",  # NGK Insulators
    "5802.T",  # Sumitomo Electric
    # Tech & Electronics
    "6758.T",  # Sony
    "6861.T",  # Keyence
    "8035.T",  # Tokyo Electron
    "6954.T",  # Fanuc
    "6702.T",  # Fujitsu
    "6752.T",  # Panasonic
    "6971.T",  # Kyocera
    "7733.T",  # Olympus
    "7751.T",  # Canon
    "4901.T",  # Fujifilm
    "4543.T",  # Terumo
    "6594.T",  # Nidec
    "6857.T",  # Advantest
    "6981.T",  # Murata Manufacturing
    "6920.T",  # Lasertec
    "6501.T",  # Hitachi
    "7731.T",  # Nikon
    "7911.T",  # Toppan
    "7912.T",  # Dai Nippon Printing
    "6841.T",  # Yokogawa Electric
    "6701.T",  # NEC
    "6103.T",  # Okuma
    # Financials
    "8306.T",  # MUFG
    "8316.T",  # SMFG
    "8411.T",  # Mizuho
    "8308.T",  # Resona
    "8604.T",  # Nomura
    "8601.T",  # Daiwa Securities
    "8766.T",  # Tokio Marine
    "8725.T",  # MS&AD Insurance
    "8630.T",  # Sompo Holdings
    "7186.T",  # Concordia Financial
    "8304.T",  # Aozora Bank
    "8331.T",  # Chiba Bank
    "8355.T",  # Shizuoka Bank
    "8354.T",  # Fukuoka Financial
    "7182.T",  # J Trust
    # Pharma
    "4502.T",  # Takeda
    "4503.T",  # Astellas
    "4507.T",  # Shionogi
    "4519.T",  # Chugai
    "4523.T",  # Eisai
    "4527.T",  # Otsuka Holdings
    "4531.T",  # Kyowa Kirin
    "4578.T",  # Otsuka -- wait this is Otsuka Corp
    # Trading / General Trading (Sogo Shosha)
    "8001.T",  # Itochu
    "8031.T",  # Mitsui & Co.
    "8058.T",  # Mitsubishi Corp
    "8053.T",  # Sumitomo Corp
    "8002.T",  # Marubeni
    "8078.T",  # Hanwa
    "8088.T",  # Iwatani
    "8060.T",  # Kanematsu
    # Food & Beverage
    "2801.T",  # Kikkoman
    "2802.T",  # Ajinomoto
    "2914.T",  # Japan Tobacco
    "2502.T",  # Asahi Group
    "2503.T",  # Kirin Holdings
    "2269.T",  # Meiji Holdings
    "2531.T",  # Takara
    "2871.T",  # Nichirei
    "2809.T",  # Kewpie
    "2810.T",  # House Foods
    "2811.T",  # Kagome
    # Chemicals
    "4063.T",  # Shin-Etsu Chemical
    "3402.T",  # Toray
    "3401.T",  # Teijin
    "3405.T",  # Kuraray
    "4042.T",  # Denka
    "4202.T",  # Daicel
    "4208.T",  # Ube Corp
    "4005.T",  # Sumitomo Chemical
    "4183.T",  # Mitsubishi Chemical
    "4188.T",  # Mitsubishi Chemical -- actually 4183
    "4185.T",  # JSR
    "4452.T",  # Kao
    "4911.T",  # Shiseido
    "4921.T",  # FANCL
    # Industrials / Machinery
    "6301.T",  # Komatsu
    "6326.T",  # Kubota
    "6367.T",  # Daikin
    "6273.T",  # SMC
    "6305.T",  # Hitachi Construction Machinery
    "6361.T",  # Ebara
    "6370.T",  # Kurita Water
    "7011.T",  # Mitsubishi Heavy
    "7012.T",  # Kawasaki Heavy
    "7013.T",  # IHI
    "7004.T",  # Hitachi Zosen
    "6113.T",  # Amada
    "6136.T",  # OSG
    "6291.T",  # Air Water
    "6479.T",  # MinebeaMitsumi
    "6471.T",  # NSK
    # Rail / Transport
    "9020.T",  # East JR
    "9021.T",  # West JR
    "9022.T",  # Central JR
    "9101.T",  # NYK Line
    "9104.T",  # Mitsui OSK
    "9107.T",  # Kawasaki Kisen
    "9201.T",  # Japan Airlines
    "9202.T",  # ANA Holdings
    "9005.T",  # Tokyu
    "9001.T",  # Tobu
    "9007.T",  # Odakyu
    "9008.T",  # Keio
    # Utilities
    "9501.T",  # Tokyo Gas (actually 9531)
    "9502.T",  # TEPCO
    "9503.T",  # Kansai Electric
    "9504.T",  # Chubu Electric
    "9505.T",  # Kyushu Electric -- wait
    "9506.T",  # Tohoku Electric
    "9507.T",  # Shikoku Electric
    "9513.T",  # J-Power
    "9531.T",  # Tokyo Gas
    "9532.T",  # Osaka Gas
    # Real Estate
    "8801.T",  # Mitsui Fudosan
    "8802.T",  # Mitsubishi Estate
    "8804.T",  # Tokyo Tatemono
    "8830.T",  # Sumitomo Realty
    # Telecom / IT Services
    "9432.T",  # NTT
    "9433.T",  # KDDI
    "9613.T",  # NTT Data
    "9434.T",  # SoftBank Corp
    "9735.T",  # Secom
    # Services / Retail
    "6098.T",  # Recruit Holdings
    "9983.T",  # Fast Retailing
    "9984.T",  # SoftBank Group
    "9843.T",  # Nitori
    "9766.T",  # Konami
    "3382.T",  # Seven & i Holdings
    "8267.T",  # AEON
    "8253.T",  # Credit Saison
    "8233.T",  # Takashimaya
    "8252.T",  # Marui
    "8028.T",  # FamilyMart (delisted?)
    "3099.T",  # Isetan Mitsukoshi
    "3086.T",  # J. Front Retailing
    # Steel / Materials
    "5401.T",  # Nippon Steel
    "5406.T",  # Kobe Steel
    "5411.T",  # JFE Holdings
    "5711.T",  # Mitsubishi Materials
    "5713.T",  # Sumitomo Metal Mining
    "5714.T",  # Dowa
    "5703.T",  # Nippon Light Metal
    "5706.T",  # Mitsui Mining & Smelting
    "5707.T",  # Toho Zinc
    # Precision / Other
    "7735.T",  # SCREEN
    "7701.T",  # Shimadzu
    "6869.T",  # Nidec -- already
    "4544.T",  # H.U. Group
    "4521.T",  # Kaken Pharma
    "4571.T",  # NanoCarrier -- too small
    # More top names to fill
    "2501.T",  # Sapporo
    "2579.T",  # Coca-Cola Bottlers Japan
    "2587.T",  # Suntory -- not listed?
    "2212.T",  # Yamazaki Baking
    "2002.T",  # Nisshin Seifun
    "2875.T",  # Nisshin Foods -- hmm
    "8007.T",  # Takashima -- actually no
    "8020.T",  # Kanematsu -- already
    "8160.T",  # Kisoji
    "7951.T",  # Yamaha
    "7956.T",  # Pigeon
    "7966.T",  # Kokuyo
    "7994.T",  # Okamura
    "5201.T",  # AGC (Asahi Glass)
    "5214.T",  # Nippon Electric Glass
    "5233.T",  # Taiheiyo Cement
    "5301.T",  # Tokai Carbon
    "5332.T",  # TOTO
    "5333.T",  # NGK -- already
    # Precision instruments
    "4540.T",  # Tsumura
    "4553.T",  # Mitsubishi Tanabe -- hmm
    "4568.T",  # Daiichi Sankyo
    "4626.T",  # Mitsubishi gas -- wait
    # Oils
    "5002.T",  # Showa Shell (idemitsu)
    "5019.T",  # Idemitsu Kosan
    "5020.T",  # ENEOS (JXTG)
    # Pulp & Paper
    "3861.T",  # Oji
    "3863.T",  # Nippon Paper
    # Mining
    "1605.T",  # INPEX
    "1662.T",  # Japan Petroleum -- not
    "1719.T",  # Hazama Ando
    # Construction
    "1801.T",  # Taisei
    "1802.T",  # Obayashi
    "1803.T",  # Shimizu
    "1812.T",  # Kajima
    "1925.T",  # Daiwa House
    "1928.T",  # Sekisui House
    # Marine
    # already covered
    # Warehousing
    "9301.T",  # Mitsubishi Logistics
    "9302.T",  # Mitsui Soko
    # Transport
    "9001.T",  # Tobu
    "9005.T",  # Tokyu
    "9007.T",  # Odakyu
    "9008.T",  # Keio
    "9009.T",  # Keisei
    "9020.T",  # JR East
    "9021.T",  # JR West
    "9022.T",  # JR Central
    # Games
    "7974.T",  # Nintendo
    "6460.T",  # Sega Sammy
    "9684.T",  # Square Enix
    "3659.T",  # Nexon
    "2432.T",  # DeNA
    "4689.T",  # Yahoo Japan (now LY Corp - 4689)
    # Retail
    "3099.T",  # Isetan Mitsukoshi
    "3086.T",  # J. Front
    "9831.T",  # Yamada Denki
    "9843.T",  # Nitori
    "9983.T",  # Fast Retailing
    "7453.T",  # Ryohin Keikaku (Muji)
]

# Deduplicate
JAPAN = list(dict.fromkeys(JAPAN))

# ── Europe ────────────────────────────────────────────────────────────────────

UK = [
    # FTSE 100 majors
    "AZN.L",   # AstraZeneca
    "HSBA.L",  # HSBC
    "SHEL.L",  # Shell
    "BP.L",    # BP
    "ULVR.L",  # Unilever
    "RIO.L",   # Rio Tinto
    "GLEN.L",  # Glencore
    "BHP.L",   # BHP Group
    "GSK.L",   # GSK
    "DGE.L",   # Diageo
    "REL.L",   # RELX
    "LSEG.L",  # LSE Group
    "RKT.L",   # Reckitt
    "IMB.L",   # Imperial Brands
    "BATS.L",  # British American Tobacco
    "NG.L",    # National Grid
    "SSE.L",   # SSE
    "RR.L",    # Rolls-Royce
    "BA.L",    # BAE Systems
    "LLOY.L",  # Lloyds
    "BARCL.L", # Barclays
    "STAN.L",  # Standard Chartered
    "PRU.L",   # Prudential
    "AV.L",    # Aviva
    "LGEN.L",  # Legal & General
    "MNG.L",   # M&G
    "EXPN.L",  # Experian
    "INF.L",   # Informa
    "ABDN.L",  # Abrdn
    "FLTR.L",  # Flutter Entertainment
    "CCH.L",   # Coca-Cola HBC
    "CRDA.L",  # Croda
    "HLMA.L",  # Halma
    "SPX.L",   # Spirax-Sarco
    "WEIR.L",  # Weir Group
    "SN.L",    # Smith & Nephew
    "FRES.L",  # Fresnillo
    "ANTO.L",  # Antofagasta
    "AAL.L",   # Anglo American
    "WPP.L",   # WPP
    "PSON.L",  # Pearson
    "SGRO.L",  # Segro
    "LAND.L",  # Land Securities
    "BLND.L",  # British Land
    "TATE.L",  # Tate & Lyle
    "RTO.L",   # Rentokil
    "ADM.L",   # Admiral
    "AUTO.L",  # Auto Trader
    "CPG.L",   # Compass
    "ENT.L",   # Entain
    "OCDO.L",  # Ocado
    "SBRY.L",  # Sainsbury
    "TSCO.L",  # Tesco
    "MKS.L",   # Marks & Spencer
    "KGF.L",   # Kingfisher
    "SMDS.L",  # DS Smith
    "JMAT.L",  # Johnson Matthey
    "MRW.L",   # Morrison (delisted?)
    "TW.L",    # Taylor Wimpey
    "PSN.L",   # Persimmon
    "BDEV.L",  # Barratt
    "DLG.L",   # Direct Line
    "SMT.L",   # Scottish Mortgage
]

GERMANY = [
    "SAP.DE",     # SAP
    "SIE.DE",     # Siemens
    "ALV.DE",     # Allianz
    "MUV2.DE",    # Munich Re
    "BAS.DE",     # BASF
    "LIN.DE",     # Linde (now US - but was on Xetra)
    "BMW.DE",     # BMW
    "VOW3.DE",    # Volkswagen
    "MBG.DE",     # Mercedes-Benz
    "BAYN.DE",    # Bayer
    "ADS.DE",     # Adidas
    "DB1.DE",     # Deutsche Boerse
    "DPW.DE",     # Deutsche Post
    "DBK.DE",     # Deutsche Bank
    "CBK.DE",     # Commerzbank
    "DTE.DE",     # Deutsche Telekom
    "EOAN.DE",    # E.ON
    "RWE.DE",     # RWE
    "IFX.DE",     # Infineon
    "FME.DE",     # Fresenius
    "MRK.DE",     # Merck KGaA
    "SHL.DE",     # Siemens Healthineers
    "SY1.DE",     # Symrise
    "HEI.DE",     # HeidelbergCement (Heidelberg Materials)
    "CON.DE",     # Continental
    "HEN3.DE",    # Henkel
    "BEI.DE",     # Beiersdorf
    "TKA.DE",     # ThyssenKrupp
    "EVK.DE",     # Evonik
    "WCH.DE",     # Wacker Chemie
    "802770.DE",  # Porsche AG (ticker: P911)
    "ZAL.DE",     # Zalando
    "MTX.DE",     # MTU Aero Engines
    "LHA.DE",     # Lufthansa
    "SZU.DE",     # Suedzucker
    "DWNI.DE",    # Deutz
    "HOT.DE",     # Hochtief
    "G24.DE",     # GEA
    "SDF.DE",     # K+S
    "VAR1.DE",    # Varta
]

FRANCE = [
    "MC.PA",     # LVMH
    "OR.PA",     # L'Oreal
    "AC.PA",     # Accor
    "AI.PA",     # Air Liquide
    "AIR.PA",    # Airbus
    "BNP.PA",    # BNP Paribas
    "CA.PA",     # Credit Agricole
    "GLE.PA",    # Societe Generale
    "CAP.PA",    # Capgemini
    "CS.PA",     # AXA
    "DG.PA",     # Vinci
    "DSY.PA",    # Dassault Systemes
    "EDF.PA",    # EDF (delisted?)
    "EL.PA",     # EssilorLuxottica
    "EN.PA",     # Bouygues
    "ENGI.PA",   # Engie
    "KER.PA",    # Kering
    "LR.PA",     # Legrand
    "MC.PA",     # LVMH -- dup
    "ML.PA",     # Michelin
    "MT.PA",     # ArcelorMittal
    "NX.PA",     # Nexity
    "POM.PA",    # Compagnie de Saint-Gobain
    "PUB.PA",    # Publicis
    "RNO.PA",    # Renault
    "SAF.PA",    # Safran
    "SAN.PA",    # Sanofi
    "SGO.PA",    # Saint-Gobain
    "STLA.PA",   # Stellantis
    "SU.PA",     # Schneider Electric
    "SW.PA",     # Sodexo
    "TEP.PA",    # Teleperformance
    "THR.PA",    # Thales
    "TTE.PA",    # TotalEnergies
    "VIE.PA",    # Veolia
    "VIV.PA",    # Vivendi
    "WLN.PA",    # Worldline
    "RMS.PA",    # Hermes
    "SEV.PA",    # Suez -- merged?
]

SWITZERLAND = [
    "NESN.SW",   # Nestle
    "NOVN.SW",   # Novartis
    "ROG.SW",    # Roche
    "UBSG.SW",   # UBS
    "CSGN.SW",   # Credit Suisse (taken over by UBS)
    "ZURN.SW",   # Zurich Insurance
    "ABBN.SW",   # ABB
    "CFR.SW",    # Richemont
    "LHN.SW",    # Lonza
    "GIVN.SW",   # Givaudan
    "SREN.SW",   # Swiss Re
    "SCMN.SW",   # Swisscom
    "ALC.SW",    # Alcon
    "SGSN.SW",   # SGS
    "GEBN.SW",   # Geberit
    "SGE.SW",    # Swiss Life
    "SNBN.SW",   # Swatch
    "SLHN.SW",   # Swiss Life -- hmm
    "BAER.SW",   # Julius Baer
    "CLN.SW",    # Clariant
    "ADEN.SW",   # Adecco
    "STMN.SW",   # Straumann
]

NETHERLANDS = [
    "ASML.AS",   # ASML
    "HEIA.AS",   # Heineken
    "INGA.AS",   # ING Group
    "PHIA.AS",   # Philips
    "AD.AS",     # ABN Amro
    "AGN.AS",    # Aegon
    "AKZA.AS",   # Akzo Nobel
    "DSM.AS",    # DSM-Firmenich
    "KPN.AS",    # KPN
    "REN.AS",    # RELX (NL listing)
    "PRX.AS",    # Prosus
    "UNA.AS",    # Unilever (NL)
    "SBMO.AS",   # SBM Offshore
    "WKL.AS",    # Wolters Kluwer
    "NN.AS",     # NN Group
    "ASRNL.AS",  # ASR Nederland
]

SPAIN = [
    "SAN.MC",    # Santander
    "BBVA.MC",   # BBVA
    "TEF.MC",    # Telefonica
    "ITX.MC",    # Inditex
    "FER.MC",    # Ferrovial
    "IBE.MC",    # Iberdrola
    "REP.MC",    # Repsol
    "ANA.MC",    # Acciona
    "ACS.MC",    # ACS
    "CABK.MC",   # CaixaBank
    "ENG.MC",    # Enagas
    "GRF.MC",    # Grifols
    "MAP.MC",    # Mapfre
    "MTS.MC",    # Cellnex
    "MEL.MC",    # Melia
    "NTGY.MC",   # Naturgy
]

ITALY = [
    "UCG.MI",    # UniCredit
    "ISP.MI",    # Intesa Sanpaolo
    "ENI.MI",    # Eni
    "ENEL.MI",   # Enel
    "STLA.MI",   # Stellantis
    "RACE.MI",   # Ferrari
    "MB.MI",     # Mediobanca
    "BMED.MI",   # Banca Mediolanum
    "BMPS.MI",   # Monte dei Paschi
    "G.MI",      # Generali
    "SPM.MI",    # Saipem
    "TEN.MI",    # Tenaris
    "TRN.MI",    # Telecom Italia
    "MONC.MI",   # Moncler
    "PRY.MI",    # Prysmian
    "CPR.MI",    # Campari
    "LDO.MI",    # Leonardo
    "REC.MI",    # Recordati
    "AMP.MI",    # Amplifon
    "DIA.MI",    # DiaSorin
    "NEO.MI",    # Neodecortech -- no
    "IP.MI",     # Interpump
    "BGN.MI",    # BREMBO
]

NORDICS = [
    # Sweden
    "VOLV-B.ST",  # Volvo
    "ERIC-B.ST",  # Ericsson
    "SEB-A.ST",   # SEB
    "SWED-A.ST",  # Swedbank
    "SHB-A.ST",   # Handelsbanken
    "HM-B.ST",    # H&M
    "ATCO-A.ST",  # Atlas Copco
    "ASSA-B.ST",  # Assa Abloy
    "SCA-B.ST",   # SCA
    "TEL2-B.ST",  # Tele2
    "TELIA.ST",   # Telia
    "NCC-B.ST",   # NCC
    "SKF-B.ST",   # SKF
    "SAND.ST",    # Sandvik
    "ALIV-SDB.ST", # Autoliv
    "BOL.ST",     # Boliden
    "ELUX-B.ST",  # Electrolux
    "GETI-B.ST",  # Getinge
    "INVE-B.ST",  # Investor
    "KINV-B.ST",  # Kinnevik
    "LATO-B.ST",  # Latour
    "LIFCO-B.ST", # Lifco
    "SECT-B.ST",  # Securitas
    "SKIS-B.ST",  # SkiStar
    "SSAB-A.ST",  # SSAB
    # Denmark
    "MAERSK-B.CO",  # Maersk
    "NOVO-B.CO",    # Novo Nordisk
    "NZYM-B.CO",    # Novozymes
    "CARR-B.CO",    # Carlsberg
    "DSV.CO",       # DSV
    "GN.CO",        # GN Store Nord
    "PNDORA.CO",    # Pandora
    "VWS.CO",       # Vestas
    "ORSTED.CO",    # Orsted
    "FLS.CO",       # FLSmidth
    "JYSK.CO",      # Jyske Bank
    "DANSKE.CO",    # Danske Bank
    # Finland
    "NOKIA.HE",   # Nokia
    "STE-R.HE",   # Stora Enso
    "UPM.HE",     # UPM
    "FORTUM.HE",  # Fortum
    "KNEBV.HE",   # Kone
    "NESTE.HE",   # Neste
    "SAMPO.HE",   # Sampo
    "TELIA1.HE",  # Telia
    "ELISA.HE",   # Elisa
    "METSO.HE",   # Metso
    "OUT1V.HE",   # Outotec
    "DNA.HE",     # DNA
    # Norway
    "EQNR.OL",    # Equinor
    "DNB.OL",     # DNB
    "NHY.OL",     # Norsk Hydro
    "TEL.OL",     # Telenor
    "YAR.OL",     # Yara
    "ORK.OL",     # Orkla
    "MOWI.OL",    # Mowi
    "AKERBP.OL",  # Aker BP
    "SUBC.OL",    # Subsea 7
    "SALM.OL",    # SalMar
]

# ── Hong Kong ─────────────────────────────────────────────────────────────────

HONG_KONG = [
    "0001.HK",  # CK Hutchison
    "0002.HK",  # CLP
    "0003.HK",  # HK & China Gas
    "0005.HK",  # HSBC
    "0006.HK",  # Power Assets
    "0008.HK",  # PCCW
    "0010.HK",  # Hang Lung
    "0012.HK",  # Henderson Land
    "0016.HK",  # SHK Properties
    "0017.HK",  # New World Development
    "0019.HK",  # Swire Pacific
    "0023.HK",  # Bank of East Asia
    "0027.HK",  # Galaxy Entertainment
    "0066.HK",  # MTR
    "0083.HK",  # Sino Land
    "0101.HK",  # Hang Lung Properties
    "0175.HK",  # Geely Auto
    "0241.HK",  # Alibaba Health
    "0267.HK",  # CITIC
    "0291.HK",  # China Resources Beer
    "0322.HK",  # Tingyi
    "0386.HK",  # Sinopec
    "0388.HK",  # HKEX
    "0669.HK",  # Techtronic
    "0688.HK",  # China Overseas Land
    "0700.HK",  # Tencent
    "0762.HK",  # China Unicom
    "0823.HK",  # Link REIT
    "0857.HK",  # PetroChina
    "0883.HK",  # CNOOC
    "0939.HK",  # CCB
    "0941.HK",  # China Mobile
    "0960.HK",  # Longfor
    "0968.HK",  # Xinyi Solar
    "0981.HK",  # SMIC
    "0992.HK",  # Lenovo
    "1038.HK",  # CK Infrastructure
    "1044.HK",  # Hengan
    "1088.HK",  # China Shenhua
    "1093.HK",  # CSPC Pharma
    "1109.HK",  # China Resources Land
    "1113.HK",  # CK Assets
    "1177.HK",  # Sino Biopharm
    "1299.HK",  # AIA
    "1398.HK",  # ICBC
    "1810.HK",  # Xiaomi
    "1876.HK",  # Budweiser APAC
    "1928.HK",  # Sands China
    "1929.HK",  # Chow Tai Fook
    "1997.HK",  # Wharf REIC
    "2018.HK",  # AAC Tech
    "2020.HK",  # ANTA Sports
    "2269.HK",  # WuXi Biologics
    "2313.HK",  # Shenzhou
    "2318.HK",  # Ping An
    "2319.HK",  # Mengniu
    "2331.HK",  # Li Ning
    "2382.HK",  # Sunny Optical
    "2388.HK",  # BOC Hong Kong
    "2628.HK",  # China Life
    "2799.HK",  # China AMC
    "3328.HK",  # Bank of Communications
    "3690.HK",  # Meituan
    "3968.HK",  # China Merchants Bank
    "3988.HK",  # Bank of China
    "6030.HK",  # CITIC Securities
    "6185.HK",  # CanSino
    "6699.HK",  # Angelalign
    "6862.HK",  # Haidilao
    "9618.HK",  # JD.com
    "9888.HK",  # Baidu
    "9999.HK",  # NetEase
    "9988.HK",  # Alibaba
]

# ── Emerging Markets ──────────────────────────────────────────────────────────

BRAZIL = [
    "ABEV3.SA",   # Ambev
    "BBAS3.SA",   # Banco do Brasil
    "BBDC3.SA",   # Bradesco
    "BBDC4.SA",   # Bradesco PN
    "BRAP4.SA",   # Bradespar
    "BRKM5.SA",   # Braskem
    "B3SA3.SA",   # B3
    "CCRO3.SA",   # CCR
    "CIEL3.SA",   # Cielo
    "CMIG4.SA",   # Cemig
    "CSAN3.SA",   # Cosan
    "CSNA3.SA",   # CSN
    "CVCB3.SA",   # CVC
    "CYRE3.SA",   # Cyrela
    "DTEX3.SA",   # Duratex
    "ELET3.SA",   # Eletrobras
    "ELET6.SA",   # Eletrobras PN
    "EMBR3.SA",   # Embraer
    "ENEV3.SA",   # Eneva
    "EQTL3.SA",   # Equatorial
    "EZTC3.SA",   # EZTEC
    "FLRY3.SA",   # Fleury
    "GGBR4.SA",   # Gerdau
    "GOAU4.SA",   # Metalurgica Gerdau
    "ITSA4.SA",   # Itausa
    "ITUB4.SA",   # Itau
    "JBSS3.SA",   # JBS
    "KLBN11.SA",  # Klabin
    "LAME4.SA",   # Lojas Americanas
    "LREN3.SA",   # Lojas Renner
    "MGLU3.SA",   # Magazine Luiza
    "MRFG3.SA",   # Marfrig
    "MRVE3.SA",   # MRV
    "MULT3.SA",   # Multiplan
    "NATU3.SA",   # Natura
    "PCAR3.SA",   # Via Varejo (ex-Pao de Acucar)
    "PETR3.SA",   # Petrobras
    "PETR4.SA",   # Petrobras PN
    "PRIO3.SA",   # PetroRio
    "RADL3.SA",   # Raia Drogasil
    "RAIL3.SA",   # Rumo Logistica
    "RENT3.SA",   # Localiza
    "SANB11.SA",  # Santander Brasil
    "SBSP3.SA",   # Sabesp
    "SULA11.SA",  # SulAmerica
    "SUZB3.SA",   # Suzano
    "TOTS3.SA",   # Totvs
    "UGPA3.SA",   # Ultrapar
    "USIM5.SA",   # Usiminas
    "VALE3.SA",   # Vale
    "VIVT3.SA",   # Telefonica Brasil
    "WEGE3.SA",   # WEG
    "YDUQ3.SA",   # Yduqs
]

INDIA = [
    "RELIANCE.NS",     # Reliance
    "TCS.NS",          # TCS
    "HDFCBANK.NS",     # HDFC Bank
    "INFY.NS",         # Infosys
    "ICICIBANK.NS",    # ICICI Bank
    "HINDUNILVR.NS",   # Hindustan Unilever
    "ITC.NS",          # ITC
    "SBIN.NS",         # SBI
    "BHARTIARTL.NS",   # Bharti Airtel
    "KOTAKBANK.NS",    # Kotak Mahindra
    "WIPRO.NS",        # Wipro
    "AXISBANK.NS",     # Axis Bank
    "BAJFINANCE.NS",   # Bajaj Finance
    "LT.NS",           # Larsen & Toubro
    "DMART.NS",        # Avenue Supermarts
    "ASIANPAINT.NS",   # Asian Paints
    "MARUTI.NS",       # Maruti Suzuki
    "TITAN.NS",        # Titan
    "SUNPHARMA.NS",    # Sun Pharma
    "HCLTECH.NS",      # HCL Tech
    "ULTRACEMCO.NS",   # UltraTech Cement
    "HINDZINC.NS",     # Hindustan Zinc
    "COALINDIA.NS",    # Coal India
    "NTPC.NS",         # NTPC
    "ONGC.NS",         # ONGC
    "POWERGRID.NS",    # Power Grid
    "TATASTEEL.NS",    # Tata Steel
    "TATAMOTORS.NS",   # Tata Motors
    "BAJAJ-AUTO.NS",   # Bajaj Auto
    "EICHERMOT.NS",    # Eicher Motors
    "HEROMOTOCO.NS",   # Hero MotoCorp
    "BRITANNIA.NS",    # Britannia
    "NESTLEIND.NS",    # Nestle India
    "TECHM.NS",        # Tech Mahindra
    "TCS.NS",          # already
    "PAGEIND.NS",      # Page Industries
    "PIDILITIND.NS",   # Pidilite
    "BAJAJFINSV.NS",   # Bajaj Finserv
    "SBILIFE.NS",      # SBI Life
    "ICICIPRULI.NS",   # ICICI Prudential
    "HDFCLIFE.NS",     # HDFC Life
    "DIVISLAB.NS",     # Divi's Labs
    "DRREDDY.NS",      # Dr Reddy's
    "CIPLA.NS",        # Cipla
    "APOLLOHOSP.NS",   # Apollo Hospitals
    "GRASIM.NS",       # Grasim
    "ADANIPORTS.NS",   # Adani Ports
    "SHREECEM.NS",     # Shree Cement
    "BPCL.NS",         # BPCL
    "IOC.NS",          # IOC
    "GAIL.NS",         # GAIL
    "M&M.NS",          # Mahindra & Mahindra
    "VEDL.NS",         # Vedanta
    "JSWSTEEL.NS",     # JSW Steel
    "INDUSINDBK.NS",   # IndusInd Bank
    "BANDHANBNK.NS",   # Bandhan Bank
    "TORNTPHARM.NS",   # Torrent Pharma
    "MARICO.NS",       # Marico
    "DABUR.NS",        # Dabur
    "COLPAL.NS",       # Colgate Palmolive
    "HAVELLS.NS",      # Havells
    "VOLTAS.NS",       # Voltas
    "SIEMENS.NS",      # Siemens India
    "ABB.NS",          # ABB India
    "BERGEPAINT.NS",   # Berger Paints
    "BIOCON.NS",       # Biocon
    "CADILAHC.NS",     # Cadila Healthcare
    "AUBANK.NS",       # AU Small Finance Bank
    "BANKBARODA.NS",   # Bank of Baroda
    "MUTHOOTFIN.NS",   # Muthoot Finance
    "PEL.NS",          # Piramal Enterprises
    "SRTRANSFIN.NS",   # Shriram Transport
    "NAUKRI.NS",       # Info Edge
    "JUBLFOOD.NS",     # Jubilant Foodworks
    "GODREJCP.NS",     # Godrej Consumer
    "GODREJIND.NS",    # Godrej Industries
    "RAMCOCEM.NS",     # Ramco Cements
    "ICICIGI.NS",      # ICICI Lombard
    "MCDOWELL-N.NS",   # United Spirits
    "CONCOR.NS",       # Container Corp
    "LUPIN.NS",        # Lupin
    "AUROPHARMA.NS",   # Aurobindo Pharma
    "ALKEM.NS",        # Alkem Labs
]

KOREA = [
    "005930.KS",  # Samsung Electronics
    "000660.KS",  # SK Hynix
    "207940.KS",  # Samsung Biologics
    "051910.KS",  # LG Chem
    "005380.KS",  # Hyundai Motor
    "006400.KS",  # Samsung SDI
    "000270.KS",  # Kia
    "068270.KS",  # Celltrion
    "105560.KS",  # KB Financial
    "055550.KS",  # Shinhan Financial
    "138040.KS",  # Meritz Financial
    "086790.KS",  # Hana Financial
    "316140.KS",  # KakaoBank
    "352820.KS",  # HYBE
    "247540.KS",  # Ecopro BM
    "003670.KS",  # Posco Holdings
    "035420.KS",  # NAVER
    "035720.KS",  # Kakao
    "036570.KS",  # NCSoft
    "251270.KS",  # Netmarble
    "112040.KS",  # HMM
    "009540.KS",  # HD Korea Shipbuilding
    "010130.KS",  # Korea Zinc
    "012330.KS",  # Hyundai Mobis
    "096770.KS",  # SK Innovation
    "034730.KS",  # SK
    "000810.KS",  # Samsung Fire & Marine
    "018260.KS",  # Samsung SDS
    "028260.KS",  # Samsung C&T
    "017670.KS",  # SK Telecom
    "030200.KS",  # KT
    "066570.KS",  # LG Electronics
    "034020.KS",  # Doosan
    "000880.KS",  # Hanwha
    "003550.KS",  # LG Corp
    "011200.KS",  # HMM -- dup?
    "139480.KS",  # E-mart
    "000720.KS",  # Hyundai Engineering
    "402340.KS",  # SK Square
    "047050.KS",  # Daewoo International
    "021240.KS",  # Coupang (not .KS, it's US)
    "377300.KS",  # KakaoPay
    "328130.KS",  # Dunamu -- not
    "323410.KS",  # KakaoBank -- dup
    "096770.KS",  # SK Innovation
    "267250.KS",  # Hyundai Heavy Industries
    "329180.KS",  # Hyundai Doosan Infracore
    "010140.KS",  # Samsung Heavy Industries
    "010620.KS",  # Hyundai Mipo
    "009830.KS",  # Hanwha Solutions
    "011070.KS",  # LG Innotek
    "033780.KS",  # KT&G
    "028050.KS",  # Samsung Engineering
    "000990.KS",  # DB Insurance
    "003490.KS",  # Korean Air
    "004990.KS",  # CJ Corp
    "097950.KS",  # CJ CheilJedang
    "271560.KS",  # Orion
    "036460.KS",  # Korea Gas Corp
    "023530.KS",  # Hanwha Aerospace
    "161390.KS",  # Hanwha Chemical -- not
    "003410.KS",  # Ssangyong Cement
    "001460.KS",  # BYC
    "008560.KS",  # Meritz -- no
]

# ── Combine ───────────────────────────────────────────────────────────────────

ALL_TICKERS = {
    "Japan":     JAPAN,
    "UK":        UK,
    "Germany":   GERMANY,
    "France":    FRANCE,
    "Switzerland": SWITZERLAND,
    "Netherlands": NETHERLANDS,
    "Spain":     SPAIN,
    "Italy":     ITALY,
    "Nordics":   NORDICS,
    "HongKong":  HONG_KONG,
    "Brazil":    BRAZIL,
    "India":     INDIA,
    "Korea":     KOREA,
}

flat = []
for grp, tickers in ALL_TICKERS.items():
    flat.extend(tickers)

# Deduplicate while preserving order
seen = set()
flat = [t for t in flat if not (t in seen or seen.add(t))]

log.info("Total tickers to download: %d", len(flat))
for grp, tickers in ALL_TICKERS.items():
    log.info("  %s: %d", grp, len(tickers))

# ── Download ──────────────────────────────────────────────────────────────────

def download_single(ticker, max_retries=5):
    """Download a single ticker with retry and exponential backoff."""
    for attempt in range(max_retries):
        try:
            data = yf.download(
                ticker,
                start=START,
                end=END,
                auto_adjust=True,
                progress=False,
            )
            if data is not None and not data.empty:
                # yfinance 1.5+ returns MultiIndex columns even for single tickers
                # e.g. ("Close", "7203.T"). Extract a proper Series.
                if isinstance(data.columns, pd.MultiIndex):
                    # Find the Close column for this ticker
                    close_cols = [c for c in data.columns if c[0] == "Close"]
                    if close_cols:
                        series = data[close_cols[0]]
                    else:
                        series = data.iloc[:, 0]
                    series = series.dropna()
                else:
                    col = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else None)
                    if col:
                        series = data[col].dropna()
                    else:
                        series = None
                if isinstance(series, pd.Series) and len(series) > 0:
                    return series
            return None
        except Exception as e:
            err_str = str(e)
            if "Rate limited" in err_str or "Too Many Requests" in err_str:
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40, 80
                log.warning("  Rate limited on %s, retry %d/%d in %ds", ticker, attempt + 1, max_retries, wait)
                time.sleep(wait)
            elif "possibly delisted" in err_str:
                log.info("  %s: possibly delisted, skipping", ticker)
                return None
            else:
                if attempt < max_retries - 1:
                    wait = 3 * (2 ** attempt)
                    log.warning("  Error on %s (attempt %d/%d): %s, retrying in %ds",
                                ticker, attempt + 1, max_retries, e, wait)
                    time.sleep(wait)
                else:
                    log.warning("  %s: failed after %d retries: %s", ticker, max_retries, e)
                    return None
    return None


def download_batch(tickers):
    """Download a batch of tickers individually. Returns dict of symbol -> Series."""
    result = {}
    for ticker in tickers:
        series = download_single(ticker)
        if series is not None:
            result[ticker] = series
        time.sleep(0.3)  # small delay between individual tickers
    return result


prices = {}  # symbol -> pd.Series (date index)
errors = []

for i in range(0, len(flat), BATCH_SIZE):
    batch = flat[i : i + BATCH_SIZE]
    log.info("Batch %d/%d: %s .. %s",
             i // BATCH_SIZE + 1, (len(flat) + BATCH_SIZE - 1) // BATCH_SIZE,
             batch[0], batch[-1])
    batch_results = download_batch(batch)
    prices.update(batch_results)
    missing = [t for t in batch if t not in batch_results]
    if missing:
        log.info("  Missing: %d/%d", len(missing), len(batch))
        errors.extend(missing)
    time.sleep(SLEEP_BETWEEN_BATCHES)

log.info("Download complete. Successfully downloaded %d / %d tickers.",
         len(prices), len(flat))

# ── Assemble DataFrame ────────────────────────────────────────────────────────

log.info("Assembling price DataFrame...")
# Filter out any non-Series entries (scalars) that may have slipped through
prices_clean = {k: v for k, v in prices.items() if isinstance(v, pd.Series) and len(v) > 1}
log.info("Clean series: %d / %d", len(prices_clean), len(prices))
df = pd.DataFrame(prices_clean)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df = df.astype("float32")

# Pretty up column names: strip suffix and organize
# Keep original ticker names as columns

log.info("Saving to %s ...", OUTPUT)
df.to_parquet(OUTPUT)
log.info("Saved: %s", OUTPUT)

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("  INTL STOCK DOWNLOAD SUMMARY")
print("=" * 72)
print(f"  Date range:       {START} to {END}")
print(f"  Total tickers:    {len(flat)}")
print(f"  Downloaded:       {len(prices)}")
print(f"  Missing/errors:   {len(errors)}")
print(f"  Output file:      {OUTPUT}")
print(f"  Shape:            {df.shape[0]} rows x {df.shape[1]} cols")
print(f"  Date range:       {df.index.min().date()} to {df.index.max().date()}")
print("-" * 72)
print("  Breakdown by market:")
for grp, tickers in ALL_TICKERS.items():
    in_data = [t for t in tickers if t in prices]
    print(f"    {grp:15s}: {len(in_data):3d} / {len(tickers):3d}")
print("-" * 72)
if errors:
    print("  Missing tickers (first 30):")
    for t in errors[:30]:
        print(f"    {t}")
    if len(errors) > 30:
        print(f"    ... and {len(errors) - 30} more")
print("=" * 72)
