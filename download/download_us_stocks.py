"""
Download as many US stocks as possible, targeting 3000+ tickers.
Saves price and volume data to parquet files.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import warnings
import io
import re
import json

warnings.filterwarnings('ignore')

print("=" * 70)
print("US STOCK UNIVERSE EXPANDER - TARGET 3000+ TICKERS")
print("=" * 70)

OUTPUT_DIR = r"D:/code/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_PARQUET = os.path.join(OUTPUT_DIR, "us_price_expanded.parquet")
VOLUME_PARQUET = os.path.join(OUTPUT_DIR, "us_volume_expanded.parquet")

START = "2000-01-01"
END   = "2024-12-31"

BATCH_SIZE   = 30
SLEEP_SEC    = 1.0
MAX_RETRIES  = 2

# ---------------------------------------------------------------------------
# 1. Wikipedia scraping helpers
# ---------------------------------------------------------------------------
def fetch_wikipedia_table(url, table_index=0):
    """Fetch a table from a Wikipedia page using pandas read_html."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        tables = pd.read_html(url, storage_options={"headers": headers})
        if tables and len(tables) > table_index:
            df = tables[table_index]
            return df
    except Exception as e:
        print(f"  [WARN] Failed to read {url}: {e}")
    return None


def extract_tickers_from_wikipedia(url, column_names=None, table_index=0):
    """Extract ticker symbols from a Wikipedia table."""
    df = fetch_wikipedia_table(url, table_index)
    if df is None:
        return set()

    tickers = set()
    # Try common column names
    candidates = column_names or ["Symbol", "Ticker", "Ticker symbol", "Symbols"]
    for col in candidates:
        if col in df.columns:
            raw = df[col].dropna().astype(str)
            for val in raw:
                val = val.strip().upper()
                # Remove exchange suffixes like .NY or .NAS
                if "." in val:
                    val = val.split(".")[0]
                # Skip non-ticker rows
                if val and not val.startswith("^") and not val.startswith("$"):
                    tickers.add(val)
            break  # found a column

    return tickers


def get_russell_3000():
    """Get Russell 3000 components from Wikipedia."""
    print("\n[1] Fetching Russell 3000 from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/Russell_3000"
    tickers = extract_tickers_from_wikipedia(url, table_index=0)
    print(f"  -> Got {len(tickers)} tickers from Russell 3000 page")
    if len(tickers) < 100:
        print("  [WARN] Russell 3000 extraction seems low; may need fallback.")
    return tickers


def get_sp500():
    """Get S&P 500 components from Wikipedia."""
    print("\n[2] Fetching S&P 500 from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tickers = extract_tickers_from_wikipedia(url, table_index=0)
    print(f"  -> Got {len(tickers)} S&P 500 tickers")
    return tickers


def get_nasdaq100():
    """Get NASDAQ-100 components from Wikipedia."""
    print("\n[3] Fetching NASDAQ-100 from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tickers = extract_tickers_from_wikipedia(url,
        column_names=["Ticker", "Symbol", "Company", "Ticker symbol"],
        table_index=0)
    print(f"  -> Got {len(tickers)} NASDAQ-100 tickers")
    if len(tickers) < 50:
        # Try alternate table
        url2 = "https://en.wikipedia.org/wiki/Nasdaq-100#Components"
        tickers2 = extract_tickers_from_wikipedia(url2, table_index=0)
        print(f"  -> Alternate table gave {len(tickers2)} tickers")
        tickers = tickers.union(tickers2)
    return tickers


# ---------------------------------------------------------------------------
# 2. Comprehensive hardcoded list of US stocks
# ---------------------------------------------------------------------------
def get_hardcoded_us_stocks():
    """
    Comprehensive list of major, mid-cap, and liquid US stocks.
    Sources: S&P 500, NASDAQ 100, Dow Jones, major ETFs, and sector leaders.
    """
    print("\n[4] Loading hardcoded list of US stocks...")

    tickers = [
        # --- DOW JONES 30 ---
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
        "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
        "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT",

        # --- S&P 500 ADDITIONAL (beyond DJI) ---
        "AAP", "ABBV", "ABMD", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM",
        "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM",
        "ALB", "ALGN", "ALK", "ALL", "ALLE", "AMAT", "AMD", "AME", "AMH", "AMT",
        "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APO", "APTV",
        "ARE", "ATO", "ATVI", "AVB", "AVGO", "AVY", "AWK", "AXON", "AXP", "AZO",
        "BAX", "BBY", "BDX", "BEN", "BF.B", "BG", "BIIB", "BIO", "BK", "BKNG",
        "BKR", "BLK", "BLL", "BMY", "BR", "BRK.B", "BRO", "BSX", "BWA", "BXP",
        "C", "CAG", "CAH", "CARR", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS",
        "CDW", "CE", "CERN", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF",
        "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP",
        "COF", "COO", "COP", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM",
        "CRWD", "CSGP", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS",
        "D", "DAL", "DAY", "DD", "DE", "DECK", "DELL", "DFS", "DG", "DGX",
        "DHI", "DHR", "DIS", "DISH", "DLR", "DLTR", "DOV", "DPZ", "DRI", "DTE",
        "DUK", "DVA", "DVN", "DXC", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX",
        "EIX", "EL", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT",
        "ERIE", "ES", "ESS", "ETN", "ETR", "ETSY", "EV", "EW", "EXC", "EXPD",
        "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV",
        "FI", "FICO", "FIS", "FITB", "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT",
        "FTV", "GD", "GE", "GEHC", "GEN", "GILD", "GIS", "GL", "GLW", "GM",
        "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS",
        "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE",
        "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM", "HWM", "IBM", "ICE",
        "IDXX", "IEX", "IFF", "INCY", "INFO", "INTC", "INTU", "IP", "IPG", "IPGP",
        "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "JBHT", "JBL", "JCI",
        "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM",
        "KKR", "KLA", "KMB", "KMI", "KMX", "KO", "KR", "KSS", "L", "LDOS",
        "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW",
        "LRCX", "LULU", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MASS", "MCD",
        "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC",
        "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR",
        "MRK", "MRNA", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU",
        "NDAQ", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG",
        "NSC", "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWSA", "NXPI", "O", "ODFL",
        "OGE", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA",
        "PAYC", "PAYX", "PCAR", "PCG", "PEAK", "PEG", "PENN", "PEP", "PFE", "PFG",
        "PG", "PGR", "PH", "PHM", "PKG", "PKI", "PLD", "PLTR", "PM", "PNC",
        "PNR", "PNW", "POOL", "PPG", "PPL", "PRGO", "PRU", "PSA", "PSX", "PTC",
        "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RHI", "RJF",
        "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RPM", "RS", "RSG", "RTX",
        "SBAC", "SBUX", "SCHW", "SHW", "SJM", "SNA", "SNAP", "SNOW", "SNPS",
        "SO", "SPG", "SPGI", "SQ", "SRE", "STE", "STLD", "STT", "STX", "STZ",
        "SWK", "SWKS", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH",
        "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP",
        "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL",
        "UAL", "UBER", "UDR", "UHS", "ULTA", "UMC", "UNH", "UNM", "UNP", "UPS",
        "URI", "USB", "V", "VLO", "VMC", "VRSK", "VRSN", "VRTX", "VST", "VTR",
        "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC",
        "WM", "WMB", "WMT", "WRB", "WSM", "WST", "WTW", "WY", "WYNN", "XEL",
        "XOM", "XRAY", "XYL", "YUM", "ZBRA", "ZBH", "ZION", "ZTS",

        # --- NASDAQ additional liquid tickers ---
        "AA", "AAON", "AAWW", "ABC", "ABCB", "ABG", "ABM", "ABR", "ACA", "ACAC",
        "ACAD", "ACEL", "ACHC", "ACHV", "ACI", "ACIC", "ACLS", "ACM", "ACMR",
        "ACVA", "ADCT", "ADEA", "ADI", "ADNT", "ADPT", "ADUS", "ADVM", "AEL",
        "AEIS", "AEM", "AEO", "AER", "AFRM", "AGCO", "AGIO", "AGL", "AGNC",
        "AGO", "AGR", "AGRO", "AGYS", "AHH", "AHR", "AHT", "AI", "AIG", "AIN",
        "AIR", "AIRC", "AIZ", "AJAX", "AJG", "AKR", "AL", "ALB", "ALC", "ALCO",
        "ALE", "ALEX", "ALG", "ALGM", "ALGN", "ALGT", "ALL", "ALLY", "ALNY",
        "ALPN", "ALRM", "ALSN", "ALT", "ALTR", "ALV", "ALX", "AM", "AMAL",
        "AMBA", "AMC", "AMCR", "AMEH", "AMG", "AMGN", "AMH", "AMKR", "AMLX",
        "AMN", "AMP", "AMPH", "AMPY", "AMR", "AMRC", "AMRX", "AMS", "AMSC",
        "AMSF", "AMSWA", "AMT", "AMTB", "AMWD", "AMZN", "AN", "ANAB", "ANDE",
        "ANF", "ANIP", "ANSS", "ANY", "AOSL", "AP", "APA", "APAM", "APD", "APG",
        "APGE", "APLS", "APO", "APP", "APPF", "APPN", "APPS", "APR", "APRE",
        "APTV", "APX", "AR", "ARAM", "ARAV", "ARCB", "ARCC", "ARCO", "ARCT",
        "ARD", "ARDX", "ARES", "ARGX", "ARI", "ARKF", "ARKG", "ARKK", "ARKQ",
        "ARKW", "ARLP", "ARM", "ARMK", "AROC", "AROW", "ARQQ", "ARQT", "ARR",
        "ARW", "ASA", "ASAI", "ASAN", "ASB", "ASGN", "ASH", "ASIX", "ASLE",
        "ASML", "ASO", "ASPN", "ASR", "ASTE", "ASTH", "ASTS", "ASUR", "ASX",
        "ATAI", "ATAT", "ATC", "ATCOL", "ATEC", "ATEN", "ATEX", "ATGE", "ATH",
        "ATHM", "ATI", "ATKR", "ATLC", "ATMU", "ATNX", "ATO", "ATOM", "ATR",
        "ATRC", "ATRO", "ATSG", "ATTO", "ATUS", "ATVI", "AUB", "AUDC", "AUR",
        "AURA", "AUUD", "AVA", "AVAH", "AVAL", "AVAV", "AVB", "AVD", "AVDL",
        "AVGO", "AVGR", "AVNS", "AVNT", "AVP", "AVPT", "AVT", "AVTR", "AVXL",
        "AVY", "AWR", "AX", "AXGN", "AXL", "AXON", "AXR", "AXS", "AXSM",
        "AXTA", "AY", "AYI", "AYX", "AZEK", "AZN", "AZPN", "AZTA", "AZZ",
        "B", "BA", "BAC", "BAH", "BALL", "BALT", "BANC", "BAND", "BANF",
        "BANR", "BANX", "BASE", "BASI", "BAT", "BATRK", "BB", "BBAI", "BBCP",
        "BBD", "BBDO", "BBIO", "BBLG", "BBU", "BBVA", "BBW", "BBY", "BC", "BCAT",
        "BCBP", "BCC", "BCE", "BCLI", "BCOR", "BCOV", "BCPC", "BCRX", "BCS",
        "BCSF", "BCTX", "BCYC", "BDC", "BDN", "BDRL", "BE", "BEAM", "BEAT",
        "BECN", "BEDU", "BEL", "BEN", "BEP", "BEPC", "BEPH", "BERY", "BEST",
        "BFAM", "BFH", "BFS", "BFST", "BG", "BGC", "BGFV", "BGH", "BGNE",
        "BGRY", "BGS", "BGSF", "BH", "BH.A", "BHAT", "BHB", "BHC", "BHF",
        "BHLB", "BHP", "BHR", "BHRB", "BHVN", "BIDU", "BIG", "BIIB", "BIL",
        "BIO", "BIOC", "BIOS", "BIRD", "BIT", "BITE", "BJ", "BJRI", "BK",
        "BKD", "BKE", "BKH", "BKI", "BKKT", "BKN", "BKSY", "BKU", "BL", "BLBD",
        "BLCO", "BLD", "BLDE", "BLDP", "BLDR", "BLFS", "BLFY", "BLK", "BLKB",
        "BLL", "BLMN", "BLND", "BLNK", "BLOK", "BLRX", "BLTE", "BLU", "BLUE",
        "BLW", "BLX", "BLZE", "BMA", "BME", "BMI", "BMO", "BMRC", "BMRN",
        "BMY", "BNED", "BNIX", "BNL", "BNNR", "BNS", "BNTC", "BNTX", "BOAT",
        "BOC", "BODY", "BOH", "BOKF", "BOLT", "BON", "BOOT", "BORR", "BOSC",
        "BOTZ", "BOW", "BOX", "BP", "BPOP", "BPRN", "BPT", "BPTH", "BPYPM",
        "BPYPN", "BPYPO", "BPYPP", "BR", "BRBR", "BRC", "BRCC", "BRD", "BRDG",
        "BRFS", "BRID", "BRK.A", "BRK.B", "BRKL", "BRKR", "BRKHU", "BRO",
        "BROS", "BRP", "BRQS", "BRRX", "BRSH", "BRSP", "BRT", "BRX", "BRY",
        "BSAC", "BSBK", "BSBR", "BSET", "BSFC", "BSIG", "BSMX", "BSPM",
        "BSQR", "BSRR", "BST", "BSTZ", "BSX", "BSY", "BTA", "BTBD", "BTBT",
        "BTCS", "BTG", "BTI", "BTO", "BTU", "BTWN", "BUD", "BUI", "BUR",
        "BURL", "BUSE", "BV", "BVN", "BVXV", "BW", "BWA", "BWEN", "BWFG",
        "BWMN", "BWXT", "BX", "BXC", "BXMT", "BXMX", "BXP", "BXSL", "BY",
        "BYD", "BYFC", "BYM", "BYND", "BZ", "BZFD", "BZH", "BZUN",
        "CACC", "CACI", "CADE", "CAE", "CAKE", "CALT", "CALX", "CAMP", "CAMT",
        "CAN", "CANF", "CANG", "CAKE", "CAPL", "CAPR", "CAR", "CARA", "CARB",
        "CARG", "CARR", "CARS", "CARV", "CASH", "CASI", "CASS", "CAT", "CATC",
        "CATO", "CATY", "CB", "CBAY", "CBB", "CBFV", "CBIO", "CBL", "CBNJ",
        "CBNK", "CBOE", "CBRE", "CBSH", "CBT", "CBU", "CBZ", "CC", "CCAP",
        "CCB", "CCBG", "CCCC", "CCCS", "CCEL", "CCI", "CCJ", "CCK", "CCL",
        "CCM", "CCNE", "CCO", "CCOI", "CCRN", "CCS", "CCU", "CCV", "CCVI",
        "CCZ", "CD", "CDAK", "CDAY", "CDE", "CDEV", "CDK", "CDLR", "CDMO",
        "CDNA", "CDNS", "CDR", "CDRE", "CDTX", "CDW", "CDXC", "CE", "CEA",
        "CECO", "CEG", "CELC", "CELH", "CELL", "CEMI", "CEN", "CENQ",
        "CENT", "CENTA", "CENX", "CEPU", "CERE", "CERO", "CERS", "CERT",
        "CET", "CETX", "CEVA", "CF", "CFB", "CFBK", "CFFE", "CFFI", "CFFN",
        "CFG", "CFIV", "CFLT", "CFR", "CFRX", "CFSB", "CG", "CGA", "CGABL",
        "CGEM", "CGEN", "CGNT", "CGNX", "CGO", "CHCI", "CHCO", "CHCT", "CHD",
        "CHE", "CHEF", "CHEK", "CHGG", "CHH", "CHI", "CHK", "CHKP", "CHMG",
        "CHN", "CHNR", "CHPT", "CHRS", "CHRW", "CHS", "CHT", "CHTR", "CHUY",
        "CHW", "CHWY", "CHX", "CI", "CIA", "CIB", "CICHY", "CIDM", "CIEN",
        "CIF", "CIG", "CIGI", "CII", "CIM", "CINC", "CINF", "CING", "CINT",
        "CIO", "CIR", "CISO", "CIT", "CIVB", "CIVI", "CIX", "CIZN", "CJJD",
        "CKPT", "CKX", "CL", "CLAR", "CLB", "CLBK", "CLBT", "CLCO", "CLDX",
        "CLEU", "CLF", "CLFD", "CLGN", "CLH", "CLIN", "CLIR", "CLLS", "CLM",
        "CLMT", "CLNE", "CLNN", "CLOE", "CLOV", "CLPR", "CLPS", "CLPT", "CLRB",
        "CLRO", "CLS", "CLSD", "CLSK", "CLSM", "CLST", "CLVR", "CLW", "CLWT",
        "CLX", "CM", "CMA", "CMBM", "CMC", "CMCL", "CMCM", "CMCO", "CMCSA",
        "CMCT", "CMD", "CME", "CMG", "CMI", "CMLS", "CMMB", "CMO", "CMP",
        "CMPO", "CMPR", "CMPS", "CMPX", "CMRE", "CMRX", "CMS", "CMSA", "CMSC",
        "CMSD", "CMT", "CMTG", "CMTL", "CNA", "CNC", "CNDT", "CNET", "CNEY",
        "CNF", "CNFR", "CNHI", "CNI", "CNK", "CNM", "CNMD", "CNNE", "CNO",
        "CNOB", "CNP", "CNQ", "CNS", "CNSL", "CNSP", "CNTA", "CNTG", "CNTQ",
        "CNTY", "CNX", "CNXC", "CNXN", "COCO", "COCP", "CODA", "CODI",
        "CODX", "COF", "COFS", "COGT", "COHU", "COIN", "COKE", "COLB", "COLD",
        "COLL", "COLM", "COMM", "COMP", "COMS", "CONE", "CONN", "CONX", "COO",
        "COOK", "COOL", "COP", "COR", "CORE", "CORR", "CORS", "CORT", "CORZ",
        "COST", "COTY", "COUP", "COUR", "COWN", "CP", "CPA", "CPAA", "CPAC",
        "CPB", "CPE", "CPF", "CPG", "CPHC", "CPHI", "CPI", "CPIX", "CPK",
        "CPLP", "CPNG", "CPOP", "CPRI", "CPRT", "CPRX", "CPS", "CPSH", "CPSI",
        "CPSS", "CPT", "CPTK", "CPUH", "CPZ", "CQP", "CR", "CRAI", "CRBP",
        "CRC", "CRCT", "CRD.A", "CRD.B", "CRDO", "CREC", "CREG", "CRESY",
        "CREX", "CRGE", "CRGY", "CRI", "CRIS", "CRK", "CRKN", "CRL", "CRM",
        "CRMD", "CRMT", "CRNC", "CRNT", "CRNX", "CRON", "CROX", "CRS", "CRSP",
        "CRSR", "CRT", "CRTO", "CRUS", "CRVO", "CRVS", "CRWD", "CRWS", "CRXT",
        "CRZN", "CS", "CSAN", "CSBR", "CSCO", "CSGP", "CSGS", "CSII", "CSIQ",
        "CSL", "CSLM", "CSLR", "CSLT", "CSMC", "CSML", "CSPI", "CSQ", "CSR",
        "CSS", "CSSE", "CSTE", "CSTL", "CSTM", "CSTR", "CSV", "CSWC", "CSWI",
        "CSX", "CTAS", "CTBI", "CTGO", "CTHR", "CTIB", "CTIC", "CTKB", "CTLP",
        "CTLT", "CTMX", "CTO", "CTOS", "CTR", "CTRA", "CTRE", "CTRM",
        "CTRN", "CTS", "CTSH", "CTSO", "CTT", "CTV", "CTVA", "CTXR", "CUBE",
        "CUBI", "CUE", "CUK", "CULP", "CUMO", "CURI", "CURO", "CURV", "CUTR",
        "CUZ", "CVBF", "CVCO", "CVCY", "CVE", "CVEO", "CVGI", "CVGW", "CVI",
        "CVII", "CVLG", "CVLT", "CVLY", "CVM", "CVNA", "CVR", "CVRX", "CVS",
        "CVU", "CVV", "CVX", "CW", "CWAN", "CWBC", "CWBR", "CWCO", "CWH",
        "CWK", "CWT", "CX", "CXAC", "CXDO", "CXE", "CXH", "CXI", "CXM",
        "CXT", "CXW", "CY", "CYAD", "CYAN", "CYB", "CYBE", "CYBN", "CYBR",
        "CYCC", "CYCN", "CYD", "CYH", "CYN", "CYRX", "CYTK", "CYTO", "CYTX",
        "CZNC", "CZOO", "CZR", "CZWI",

        # --- Mid-cap and others ---
        "DADA", "DAIO", "DAKT", "DAL", "DAN", "DAO", "DAR", "DARE", "DASH",
        "DAVA", "DAVE", "DB", "DBD", "DBI", "DBL", "DBRG", "DBTX", "DBVT",
        "DBX", "DCBO", "DCF", "DCFC", "DCI", "DCO", "DCOM", "DCP", "DCPH",
        "DCT", "DCTH", "DCU", "DCUE", "DD", "DDD", "DDOG", "DDS", "DE", "DEA",
        "DEC", "DECK", "DEI", "DELL", "DEN", "DENN", "DEO", "DERM", "DESP",
        "DFH", "DFIN", "DFLI", "DFP", "DFS", "DG", "DGICA", "DGII", "DGLY",
        "DGP", "DGRS", "DGRW", "DGS", "DGT", "DH", "DHC", "DHI", "DHIL",
        "DHR", "DHT", "DHX", "DIAL", "DIAX", "DIGA", "DIGB", "DIGS", "DIM",
        "DIN", "DIOD", "DIS", "DISA", "DISH", "DK", "DKL", "DKS", "DLA",
        "DLB", "DLHC", "DLN", "DLO", "DLPN", "DLR", "DLTH", "DLTR", "DLX",
        "DM", "DMA", "DMAC", "DMLP", "DMO", "DMRC", "DMS", "DMTK", "DMYS",
        "DNA", "DNB", "DNK", "DNP", "DNUT", "DO", "DOC", "DOCN", "DOCS",
        "DOCU", "DOGZ", "DOMA", "DOMO", "DOOO", "DORM", "DOUG", "DOV", "DOW",
        "DOX", "DOYU", "DPCS", "DPG", "DPRO", "DPS", "DPZ", "DQ", "DRCT",
        "DRD", "DRH", "DRI", "DRIO", "DRMA", "DRQ", "DRRX", "DRS", "DRVN",
        "DS", "DSE", "DSGR", "DSGX", "DSI", "DSKE", "DSL", "DSM", "DSP",
        "DSS", "DSU", "DSWL", "DSX", "DT", "DTB", "DTC", "DTE", "DTF",
        "DTG", "DTH", "DTIL", "DTM", "DTOC", "DTSS", "DTST", "DUK", "DUKB",
        "DUOL", "DURA", "DUSA", "DVA", "DVAX", "DVN", "DWAC", "DWCR", "DWF",
        "DWSN", "DWX", "DX", "DXC", "DXCM", "DXF", "DXLG", "DXPE", "DXR",
        "DXYN", "DY", "DYAI", "DYN", "DYNF", "DYNT", "DZSI",

        # More mid caps E-H
        "EAF", "EBAY", "EBC", "EBF", "EBIX", "EBMT", "EBON", "EBTC", "EC", "ECAT",
        "ECL", "ECOL", "ECOM", "ECPG", "ECVT", "ED", "EDAP", "EDBL", "EDC", "EDD",
        "EDEN", "EDF", "EDI", "EDIT", "EDN", "EDR", "EDRY", "EDSA", "EDT", "EDUC",
        "EDV", "EDVT", "EE", "EEA", "EEFT", "EEIQ", "EELV", "EEM", "EEMA", "EEMD",
        "EEMS", "EEMV", "EEMX", "EES", "EET", "EEV", "EEX", "EFA", "EFAD", "EFAS",
        "EFAV", "EFF", "EFG", "EFIV", "EFIX", "EFNL", "EFO", "EFOI", "EFR", "EFSC",
        "EFSH", "EFT", "EFTR", "EFV", "EFX", "EGAN", "EGBN", "EGC", "EGF", "EGGF",
        "EGHT", "EGIO", "EGIS", "EGLE", "EGO", "EGP", "EGRX", "EGY", "EHC", "EHI",
        "EHTH", "EIC", "EICA", "EIDX", "EIG", "EIGR", "EIM", "EINC", "EIRL", "EIS",
        "EIX", "EJH", "EKAR", "EKG", "EKSO", "EL", "ELA", "ELAN", "ELAT", "ELC",
        "ELDN", "ELEV", "ELF", "ELME", "ELOX", "ELP", "ELS", "ELSE", "ELTK",
        "ELV", "ELYS", "EM", "EMAN", "EMB", "EMBC", "EMBID", "EMCB", "EMCC",
        "EMCG", "EMC", "EMD", "EME", "EMF", "EMFM", "EMGC", "EMGF", "EMHC",
        "EMHY", "EMIF", "EMKR", "EML", "EMLD", "EMLP", "EMN", "EMO",
        "EMP", "EMQQ", "EMR", "EMSG", "EMTL", "EMTY", "EMX", "EMXC", "ENB",
        "ENCP", "ENDP", "ENER", "ENFN", "ENG", "ENGN", "ENIC", "ENJ", "ENLC",
        "ENLT", "ENNV", "ENOB", "ENOR", "ENOV", "ENPH", "ENR", "ENS", "ENSC",
        "ENSG", "ENSV", "ENTA", "ENTG", "ENTX", "ENV", "ENVA", "ENVB", "ENVX",
        "ENX", "ENZ", "EOCT", "EOG", "EOLS", "EOSE", "EP", "EPAC", "EPAM",
        "EPAY", "EPC", "EPD", "EPHE", "EPI", "EPIX", "EPM", "EPOL", "EPP",
        "EPR", "EPRE", "EPRF", "EPRT", "EPS", "EPSN", "EPU", "EPWR", "EPZM",
        "EQ", "EQAL", "EQBK", "EQC", "EQH", "EQIN", "EQIX", "EQNR", "EQR",
        "EQRX", "EQS", "EQT", "EQWL", "EQX", "ERAS", "ERES", "ERF", "ERH",
        "ERIC", "ERIE", "ERII", "ERJ", "ERM", "ERO", "ERS", "ERX", "ERY",
        "ES", "ESAB", "ESAC", "ESBA", "ESCA", "ESE", "ESEA", "ESGC", "ESGD",
        "ESGE", "ESGR", "ESGS", "ESGU", "ESGV", "ESHY", "ESI", "ESIX", "ESLT",
        "ESML", "ESMT", "ESNT", "ESP", "ESPO", "ESPR", "ESQ", "ESRT", "ESS",
        "ESSA", "ESSC", "ESTA", "ESTC", "ESTE", "ESTO", "ET", "ETB", "ETD",
        "ETG", "ETHO", "ETJ", "ETN", "ETO", "ETON", "ETR", "ETRN", "ETS",
        "ETSY", "ETV", "ETW", "ETWO", "ETX", "ETY", "EUDA", "EUFN", "EUM",
        "EUO", "EUSC", "EV", "EVA", "EVAA", "EVAV", "EVBG", "EVBN", "EVC",
        "EVCM", "EVE", "EVEN", "EVER", "EVEX", "EVF", "EVG", "EVGN", "EVGO",
        "EVH", "EVI", "EVLO", "EVLV", "EVM", "EVN", "EVO", "EVOK", "EVR",
        "EVRG", "EVRI", "EVT", "EVTC", "EVTL", "EVTV", "EVV", "EW", "EWBC",
        "EWCO", "EWCZ", "EWEB", "EWEQ", "EWG", "EWGS", "EWI", "EWJ", "EWK",
        "EWL", "EWM", "EWMC", "EWN", "EWO", "EWP", "EWQ", "EWRE", "EWS",
        "EWSC", "EWT", "EWTX", "EWU", "EWV", "EWX", "EWY", "EWZ", "EXAI",
        "EXAS", "EXC", "EXCEL", "EXD", "EXE", "EXEL", "EXFY", "EXG", "EXI",
        "EXK", "EXLS", "EXP", "EXPD", "EXPE", "EXPI", "EXPO", "EXR", "EXTR",
        "EYE", "EYEN", "EYPT", "EZA", "EZPW",

        # F
        "F", "FAB", "FACT", "FAD", "FAF", "FALN", "FAM", "FAMI", "FANG",
        "FANH", "FAPR", "FARO", "FAS", "FAST", "FAT", "FATBB", "FATBP", "FATH",
        "FATZ", "FAX", "FB", "FBAL", "FBAN", "FBC", "FBHS", "FBIO", "FBIOP",
        "FBK", "FBL", "FBMS", "FBNC", "FBND", "FBP", "FBRT", "FBT", "FBTC",
        "FBZ", "FC", "FCA", "FCAL", "FCAP", "FCAX", "FCBC", "FCCO", "FCEF",
        "FCEL", "FCF", "FCFS", "FCG", "FCLD", "FCMY", "FCN", "FCNCA", "FCNCO",
        "FCO", "FCOM", "FCOR", "FCPI", "FCPT", "FCRD", "FCRX", "FCSH", "FCT",
        "FCTR", "FCUS", "FCVT", "FCX", "FDBC", "FDD", "FDEF", "FDEM", "FDEU",
        "FDEW", "FDG", "FDHY", "FDIG", "FDIS", "FDIV", "FDL", "FDLO", "FDLS",
        "FDM", "FDMO", "FDMT", "FDN", "FDNJ", "FDP", "FDRV", "FDS", "FDT",
        "FDTS", "FDUS", "FDV", "FDVV", "FDX", "FE", "FEDL", "FEDU", "FEI",
        "FEIM", "FELE", "FEM", "FEMB", "FEMS", "FEMY", "FEN", "FENC", "FENG",
        "FEO", "FERG", "FET", "FEX", "FEXD", "FEZ", "FF", "FFA", "FFBC",
        "FFC", "FFEB", "FFHG", "FFIC", "FFIN", "FFIU", "FFIV", "FFND", "FFNW",
        "FFSB", "FFSG", "FFTG", "FFTI", "FFTY", "FG", "FGB", "FGBI", "FGBIP",
        "FGCF", "FGD", "FGEN", "FGF", "FGI", "FGL", "FGM", "FGN", "FHB",
        "FHI", "FHLC", "FHLT", "FHN", "FHTX", "FI", "FIBK", "FICO", "FICV",
        "FID", "FIDI", "FIDU", "FIEE", "FIF", "FIGS", "FIHD", "FILL", "FINS",
        "FINV", "FIP", "FIS", "FISI", "FISK", "FISV", "FITB", "FITBI", "FITBO",
        "FITBP", "FIVA", "FIVE", "FIVN", "FIW", "FIX", "FIXD", "FIXX", "FIZZ",
        "FJAN", "FJP", "FJUL", "FJUN", "FKU", "FKYI", "FL", "FLAG", "FLBH",
        "FLC", "FLCO", "FLDR", "FLDZ", "FLEX", "FLFE", "FLFV", "FLGC", "FLGT",
        "FLIC", "FLIN", "FLJ", "FLL", "FLLV", "FLM", "FLME", "FLMN", "FLNC",
        "FLNG", "FLNQ", "FLO", "FLOT", "FLQL", "FLQM", "FLQS", "FLR", "FLRG",
        "FLRN", "FLRT", "FLS", "FLSA", "FLSP", "FLT", "FLTB", "FLTN", "FLTR",
        "FLUD", "FLUX", "FLWS", "FLXS", "FLYD", "FLYW", "FLZA", "FM", "FMAG",
        "FMAO", "FMAR", "FMAT", "FMB", "FMBI", "FMBK", "FMC", "FMET", "FMF",
        "FMHI", "FMIL", "FMIV", "FMK", "FMN", "FMNB", "FMO", "FMS", "FMX",
        "FMY", "FN", "FNA", "FNB", "FNCL", "FND", "FNDA", "FNDB", "FNDC",
        "FNDF", "FNDX", "FNF", "FNFI", "FNGD", "FNGG", "FNGO", "FNGR", "FNGS",
        "FNGU", "FNK", "FNKO", "FNLC", "FNMA", "FNMBI", "FNOX", "FNV", "FNVM",
        "FNWB", "FNWD", "FOA", "FOCT", "FOLD", "FONR", "FOO", "FOR", "FORA",
        "FORD", "FORG", "FORH", "FORM", "FORR", "FORTY", "FOSL", "FOUR", "FOURR",
        "FOX", "FOXA", "FOXF", "FOXO", "FPA", "FPAC", "FPAY", "FPE", "FPEI",
        "FPF", "FPH", "FPI", "FPL", "FPRO", "FPX", "FQAL", "FR", "FRA",
        "FRAF", "FRAY", "FRBA", "FRBK", "FRC", "FRD", "FREL", "FREQ", "FRES",
        "FREY", "FRG", "FRGE", "FRGI", "FRGT", "FRHC", "FRI", "FRL", "FRLA",
        "FRME", "FRO", "FROM", "FROO", "FRPH", "FRPT", "FRSH", "FRST", "FRSX",
        "FRT", "FRTZ", "FRXB", "FSBC", "FSBW", "FSCO", "FSD", "FSEA", "FSEC",
        "FSEP", "FSFG", "FSI", "FSIG", "FSK", "FSLR", "FSLY", "FSM", "FSMD",
        "FSMO", "FSNB", "FSP", "FSRD", "FSRX", "FSS", "FSSI", "FSTA", "FSTR",
        "FSTX", "FSV", "FSZ", "FT", "FTA", "FTAG", "FTAI", "FTBD", "FTBF",
        "FTBI", "FTC", "FTCA", "FTCH", "FTCI", "FTCS", "FTDR", "FTDS", "FTEC",
        "FTEK", "FTGC", "FTHI", "FTHM", "FTHY", "FTI", "FTK", "FTLS", "FTNT",
        "FTS", "FTSL", "FTSM", "FTV", "FTXH", "FTXL", "FTXN", "FTXO", "FTXW",
        "FUBO", "FUL", "FULC", "FULT", "FULTP", "FUMB", "FUN", "FUNC", "FUND",
        "FUNL", "FURY", "FUSB", "FUSN", "FUTU", "FUTY", "FUV", "FV", "FVAL",
        "FVC", "FVCB", "FVD", "FVL", "FVRR", "FWAC", "FWBI", "FWED", "FWONA",
        "FWONK", "FWRD", "FWRG", "FXC", "FXE", "FXF", "FXG", "FXH", "FXL",
        "FXN", "FXNC", "FXO", "FXP", "FXR", "FXTB", "FXU", "FXY", "FXZ",
        "FYBR", "FYC", "FYLD", "FYT", "FYX",

        # G
        "G", "GAA", "GAB", "GABC", "GABF", "GAIA", "GAIN", "GAINL", "GAL",
        "GALT", "GAM", "GAMB", "GAMR", "GAN", "GANX", "GAQ", "GATA", "GATO",
        "GATX", "GAU", "GAW", "GBCI", "GBDC", "GBIO", "GBLI", "GBLIL", "GBNH",
        "GBNY", "GBR", "GBRG", "GBS", "GBTG", "GBX", "GCBC", "GCC", "GCI",
        "GCMG", "GCO", "GCOW", "GCP", "GCRE", "GCT", "GCTK", "GCV", "GD",
        "GDDY", "GDEN", "GDEV", "GDI", "GDL", "GDO", "GDOT", "GDRX", "GDS",
        "GDV", "GDYN", "GE", "GECC", "GECCM", "GECCO", "GEEX", "GEF", "GEG",
        "GEGGL", "GEHC", "GEI", "GEIG", "GEL", "GEN", "GENC", "GENE", "GENI",
        "GENK", "GEO", "GEOS", "GER", "GERM", "GERN", "GES", "GET", "GETY",
        "GEVO", "GF", "GFAI", "GFF", "GFG", "GFGD", "GFGF", "GFL", "GFOF",
        "GFS", "GFX", "GGAL", "GGB", "GGE", "GGG", "GGN", "GGR", "GGRW",
        "GGT", "GGZ", "GH", "GHC", "GHG", "GHI", "GHL", "GHLD", "GHM",
        "GHRS", "GHSI", "GHY", "GI", "GIA", "GIB", "GIC", "GIFI", "GIG",
        "GIGB", "GIGM", "GIGSE", "GII", "GIII", "GIL", "GILD", "GILT", "GIM",
        "GINN", "GIPR", "GIS", "GIX", "GJH", "GJO", "GJP", "GJR", "GJS",
        "GJT", "GKOS", "GL", "GLAD", "GLADL", "GLAQ", "GLBE", "GLBS",
        "GLBZ", "GLC", "GLD", "GLDD", "GLDG", "GLDM", "GLDX", "GLEE", "GLG",
        "GLIN", "GLL", "GLLI", "GLM", "GLMD", "GLNG", "GLO", "GLOB", "GLOG",
        "GLOP", "GLP", "GLPG", "GLPI", "GLQ", "GLRE", "GLRY", "GLS", "GLSI",
        "GLT", "GLTA", "GLTO", "GLU", "GLUE", "GLV", "GLW", "GLYC", "GM",
        "GMAB", "GMBL", "GMDA", "GME", "GMED", "GMF", "GMFIU", "GMGI", "GMII",
        "GMK", "GMOM", "GMRE", "GMS", "GMVD", "GMWX", "GN", "GNE", "GNFG",
        "GNK", "GNL", "GNLN", "GNOM", "GNPX", "GNR", "GNRC", "GNS", "GNSS",
        "GNT", "GNTA", "GNTX", "GNTY", "GNU", "GNUS", "GNW", "GO", "GOAU",
        "GOCO", "GODN", "GOEV", "GOGL", "GOGN", "GOL", "GOLD", "GOLF", "GOOD",
        "GOODN", "GOODO", "GOOG", "GOOGL", "GOOS", "GORO", "GOSS", "GOTU",
        "GOVT", "GP", "GPAC", "GPC", "GPI", "GPK", "GPL", "GPOR", "GPP",
        "GPRE", "GPRO", "GPS", "GRAB", "GRAY", "GRBK", "GRC", "GRCL", "GRDI",
        "GREEL", "GRF", "GRFS", "GRFX", "GRID", "GRIF", "GRIL", "GRIN", "GRMN",
        "GRN", "GRNA", "GRNB", "GRND", "GRNQ", "GRNR", "GROW", "GRPH", "GRPN",
        "GRR", "GRTS", "GRTX", "GRVY", "GRWG", "GRX", "GS", "GSAT", "GSBC",
        "GSBD", "GSD", "GSF", "GSHD", "GSIE", "GSIT", "GSIW", "GSK", "GSL",
        "GSLC", "GSM", "GSMG", "GSP", "GSPI", "GSQB", "GSS", "GSST", "GST",
        "GSUN", "GT", "GTE", "GTEC", "GTH", "GTHX", "GTIM", "GTLB", "GTLS",
        "GTN", "GTO", "GTP", "GTPA", "GTR", "GTX", "GTY", "GUG", "GUID",
        "GULF", "GUNR", "GURE", "GURU", "GUSH", "GUT", "GVA", "GPAC", "GVP",
        "GWB", "GWGH", "GWH", "GWRE", "GWRS", "GWW", "GXO", "GYLD", "GYRO",

        # H
        "H", "HA", "HACK", "HAE", "HAFC", "HAIA", "HAIN", "HAL", "HALL",
        "HALO", "HAO", "HAP", "HAPP", "HAQ", "HARP", "HART", "HAS", "HASI",
        "HAUD", "HAUS", "HAWX", "HAYN", "HAYW", "HBAN", "HBB", "HBANM",
        "HBANP", "HBAPL", "HBCP", "HBI", "HBIO", "HBM", "HBNC", "HBT", "HCA",
        "HCAR", "HCAT", "HCC", "HCCI", "HCDI", "HCDIP", "HCDIZ", "HCI",
        "HCKT", "HCM", "HCMA", "HCMF", "HCNE", "HCOM", "HCP", "HCRI", "HCSG",
        "HCTI", "HCVI", "HCWB", "HD", "HDA", "HDB", "HDEF", "HDG", "HDGE",
        "HDIF", "HDIV", "HDL", "HDSN", "HDUS", "HDV", "HE", "HEAR", "HEDJ",
        "HEEM", "HEES", "HEI", "HELE", "HELO", "HEP", "HEPA", "HERA", "HERD",
        "HERO", "HES", "HESM", "HEWC", "HEWG", "HEWJ", "HEWL", "HEWU",
        "HEXO", "HEZU", "HFBL", "HFFG", "HFGO", "HFRO", "HFWA", "HGBL",
        "HGEN", "HGER", "HGLB", "HGTY", "HGV", "HHGC", "HHH", "HHS", "HI",
        "HIBB", "HIBL", "HIBS", "HIDE", "HIE", "HIFS", "HIG", "HIGH", "HIHO",
        "HII", "HIL", "HIMS", "HIMX", "HIO", "HIPO", "HIPS", "HISF", "HITI",
        "HIVE", "HIW", "HIX", "HIYO", "HKND", "HL", "HLF", "HLGN", "HLI",
        "HLIT", "HLLY", "HLMN", "HLN", "HLNE", "HLP", "HLT", "HLTH", "HLVX",
        "HLX", "HMA", "HMC", "HMN", "HMNF", "HMNY", "HMPT", "HMST", "HMY",
        "HNDL", "HNI", "HNH", "HNNA", "HNP", "HNRG", "HNST", "HNW", "HOFT",
        "HOFV", "HOG", "HOLD", "HOLI", "HOLX", "HOME", "HOMB", "HON", "HONE",
        "HONR", "HOOD", "HOOK", "HOPE", "HOTH", "HOUR", "HOUS", "HOWL", "HP",
        "HPE", "HPF", "HPI", "HPK", "HPLT", "HPP", "HPQ", "HPS", "HQH",
        "HQI", "HQL", "HQY", "HR", "HRB", "HRI", "HRL", "HRMY", "HROW",
        "HRT", "HRTG", "HRTS", "HRZN", "HSAI", "HSBC", "HSC", "HSCS", "HSCZ",
        "HSDEC", "HSDT", "HSIC", "HSII", "HSKA", "HSLV", "HSMV", "HSON",
        "HSPO", "HST", "HSTM", "HSTO", "HSUN", "HSY", "HT", "HTA", "HTBI",
        "HTBK", "HTCR", "HTD", "HTFB", "HTFC", "HTGC", "HTGM", "HTH", "HTHT",
        "HTIA", "HTIBP", "HTLD", "HTLF", "HTLFP", "HTOO", "HTRB", "HTUS",
        "HTY", "HTZ", "HUBB", "HUBG", "HUBS", "HUDI", "HUGE", "HUIZ", "HUM",
        "HUMA", "HUN", "HURC", "HURN", "HUSV", "HUT", "HUYA", "HVBC", "HVT",
        "HWBK", "HWC", "HWEL", "HWKN", "HWM", "HXL", "HY", "HYB", "HYBL",
        "HYDB", "HYDW", "HYEM", "HYG", "HYGH", "HYGI", "HYGO", "HYGV",
        "HYHG", "HYI", "HYIN", "HYLB", "HYLD", "HYLS", "HYLV", "HYMB",
        "HYMC", "HYPR", "HYRM", "HYS", "HYT", "HYTR", "HYUP", "HYW",
        "HYXF", "HYXU", "HZN", "HZO",

        # I
        "IAA", "IAC", "IAE", "IAF", "IAG", "IAGG", "IAI", "IAK", "IAPR",
        "IART", "IAT", "IAU", "IAUF", "IAUM", "IAUX", "IBA", "IBB", "IBBQ",
        "IBCE", "IBCP", "IBD", "IBDC", "IBDO", "IBDR", "IBEX", "IBHC",
        "IBHD", "IBHE", "IBHF", "IBIO", "IBKR", "IBM", "IBN", "IBND",
        "IBOC", "IBP", "IBRX", "IBSL", "IBTA", "IBTB", "IBTD", "IBTE", "IBTF",
        "IBTG", "IBTH", "IBTI", "IBTJ", "IBTK", "IBTL", "IBTM", "IBN",
        "ICAD", "ICAP", "ICCC", "ICCH", "ICCM", "ICD", "ICE", "ICF", "ICFI",
        "ICG", "ICHR", "ICL", "ICLK", "ICLN", "ICMB", "ICOL", "ICOW", "ICR",
        "ICSH", "ICUI", "ICVX", "ID", "IDA", "IDAI", "IDAT", "IDCC", "IDE",
        "IDEV", "IDEX", "IDGT", "IDHD", "IDHQ", "IDIV", "IDLB", "IDLV",
        "IDME", "IDMO", "IDN", "IDNA", "IDOG", "IDR", "IDRA", "IDRV",
        "IDT", "IDU", "IDV", "IDW", "IDX", "IDXX", "IDYA", "IEDI", "IEF",
        "IEMG", "IESC", "IETC", "IEUR", "IEUS", "IEV", "IEX", "IEZ", "IF",
        "IFBD", "IFF", "IFIN", "IFMK", "IFN", "IFRX", "IFS", "IG", "IGA",
        "IGBH", "IGC", "IGD", "IGE", "IGEB", "IGF", "IGHG", "IGI", "IGIB",
        "IGID", "IGIH", "IGLB", "IGLD", "IGM", "IGMS", "IGOV", "IGR",
        "IGRO", "IGSB", "IGT", "IGTA", "IGV", "IH", "IHAK", "IHD", "IHDG",
        "IHE", "IHF", "IHG", "IHI", "IHIT", "IHRT", "IHS", "IHT", "IIC",
        "IIF", "IIGD", "IIGV", "III", "IIIN", "IIIV", "IINN", "IIP",
        "IIPR", "IIVI", "IJAN", "IJH", "IJS", "IJT", "IJUL", "IKNA",
        "IKNX", "ILAG", "ILCG", "ILCV", "ILF", "ILMN", "ILTB", "IMAB",
        "IMAC", "IMAQ", "IMAX", "IMBI", "IMBIL", "IMCB", "IMCC", "IMCG",
        "IMCR", "IMCV", "IMGN", "IMGO", "IMKTA", "IMLP", "IMMP", "IMMR",
        "IMMX", "IMNM", "IMO", "IMOM", "IMOS", "IMPL", "IMPP", "IMPPP",
        "IMRN", "IMRX", "IMTB", "IMTE", "IMTM", "IMTX", "IMUX", "IMV",
        "IMVT", "IMXI", "INAB", "INAQ", "INBK", "INBX", "INCO", "INCR",
        "INCY", "INDA", "INDB", "INDF", "INDI", "INDL", "INDS", "INDT",
        "INDY", "INF", "INFA", "INFI", "INFL", "INFN", "INFU", "INFY",
        "ING", "INGN", "INGR", "INKA", "INKM", "INLX", "INMD", "INN",
        "INNO", "INNV", "INO", "INOD", "INPX", "INR", "INS", "INSG",
        "INSI", "INSM", "INSP", "INST", "INSW", "INT", "INTA", "INTC",
        "INTE", "INTF", "INTG", "INTJ", "INTR", "INTT", "INTU", "INTX",
        "INUV", "INVA", "INVE", "INVH", "INVO", "INVZ", "INZY", "IOBT",
        "IONQ", "IONS", "IOO", "IOR", "IOSP", "IOT", "IOVA", "IP", "IPA",
        "IPAD", "IPAR", "IPB", "IPDN", "IPG", "IPGP", "IPHA", "IPI",
        "IPKW", "IPO", "IPOS", "IPPP", "IPSC", "IPW", "IPWR", "IQ", "IQI",
        "IQM", "IQMD", "IQSI", "IQSU", "IQV", "IR", "IRBA", "IRBO", "IRBT",
        "IRDM", "IREN", "IRIX", "IRM", "IRMD", "IRON", "IROQ", "IRS",
        "IRT", "IRTC", "IRWD", "ISCB", "ISCF", "ISD", "ISDR", "ISDS",
        "ISDX", "ISEE", "ISEM", "ISG", "ISHP", "ISIG", "ISMD", "ISNS",
        "ISPC", "ISPO", "ISR", "ISRA", "ISRG", "ISRL", "ISSC", "ISTB",
        "ISTR", "ISUN", "IT", "ITA", "ITAN", "ITB", "ITCI", "ITCL",
        "ITCN", "ITDB", "ITE", "ITGR", "ITI", "ITIC", "ITM", "ITOS",
        "ITOT", "ITP", "ITRG", "ITRI", "ITRM", "ITRN", "ITT", "ITUB",
        "ITW", "IUS", "IUSB", "IUSG", "IUSS", "IUSV", "IVAL", "IVBC",
        "IVCA", "IVDA", "IVDV", "IVE", "IVEG", "IVH", "IVLC", "IVLU",
        "IVOG", "IVOL", "IVOO", "IVOP", "IVR", "IVRA", "IVSG", "IVT",
        "IVV", "IVW", "IVZ", "IWB", "IWC", "IWD", "IWF", "IWL", "IWM",
        "IWN", "IWO", "IWP", "IWR", "IWS", "IWV", "IWX", "IWY", "IX",
        "IXAQ", "IXC", "IXG", "IXJ", "IXN", "IXP", "IXSE", "IXUS", "IYC",
        "IYE", "IYF", "IYG", "IYH", "IYJ", "IYK", "IYR", "IYT", "IYW",
        "IYY", "IYZ", "IZEA", "IZRL",
    ]

    print(f"  -> Hardcoded list has {len(tickers)} tickers")
    return set(tickers)


# ---------------------------------------------------------------------------
# 3. Try to get all NYSE and NASDAQ listed stocks via NASDAQ Trader
# ---------------------------------------------------------------------------
def get_nasdaqtrader_list():
    """Fetch the full list of NASDAQ and NYSE listed stocks from NASDAQ Trader."""
    print("\n[5] Fetching NASDAQ/NYSE listed stocks from NASDAQ Trader...")
    tickers = set()

    urls = [
        ("NASDAQ", "ftp://ftp.nasdaqtrader.com/symboldirectory/nasdaqlisted.txt"),
        ("NYSE",   "ftp://ftp.nasdaqtrader.com/symboldirectory/otherlisted.txt"),
    ]

    for exchange, url in urls:
        try:
            import urllib.request
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("utf-8")
            lines = text.splitlines()
            count = 0
            for line in lines:
                if "|" in line and not line.startswith("Symbol"):
                    symbol = line.split("|")[0].strip()
                    if symbol and symbol.isalpha() and symbol.isascii():
                        tickers.add(symbol.upper())
                        count += 1
            print(f"  -> Got {count} tickers from {exchange} ({len(tickers)} unique so far)")
        except Exception as e:
            print(f"  [WARN] Failed to fetch {exchange} list: {e}")
            print(f"  [INFO] Trying HTTP fallback for {exchange}...")
            try:
                http_url = url.replace("ftp://", "https://")
                req = urllib.request.Request(http_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    text = resp.read().decode("utf-8")
                lines = text.splitlines()
                count = 0
                for line in lines:
                    if "|" in line and not line.startswith("Symbol"):
                        symbol = line.split("|")[0].strip()
                        if symbol and symbol.isalpha() and symbol.isascii():
                            tickers.add(symbol.upper())
                            count += 1
                print(f"  -> Got {count} tickers from {exchange} (HTTP)")
            except Exception as e2:
                print(f"  [WARN] HTTP fallback also failed: {e2}")

    return tickers


# ---------------------------------------------------------------------------
# 4. Alternative: download from BATS/NYSE via web
# ---------------------------------------------------------------------------
def get_bats_list():
    """Try to get BATS listed symbols."""
    print("\n[6] Fetching symbols from BATS exchange...")
    tickers = set()
    try:
        import urllib.request
        url = "https://www.batstrading.com/market_data/symbol_list/symbol_list.csv"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            text = resp.read().decode("utf-8")
        for line in text.splitlines():
            if "," in line and "Symbol" not in line:
                symbol = line.split(",")[0].strip().upper()
                if symbol and symbol.isalpha() and symbol.isascii():
                    tickers.add(symbol)
        print(f"  -> Got {len(tickers)} symbols from BATS")
    except Exception as e:
        print(f"  [WARN] BATS list unavailable: {e}")
    return tickers


# ---------------------------------------------------------------------------
# 5. Collect all tickers
# ---------------------------------------------------------------------------
def collect_all_tickers():
    """Aggregate tickers from all sources, deduplicate, clean."""
    all_tickers = set()

    # Try each source
    sources = [
        ("Russell 3000", get_russell_3000),
        ("S&P 500", get_sp500),
        ("NASDAQ-100", get_nasdaq100),
        ("Hardcoded", get_hardcoded_us_stocks),
        ("NASDAQ Trader", get_nasdaqtrader_list),
        ("BATS", get_bats_list),
    ]

    for name, func in sources:
        try:
            tickers = func()
            all_tickers.update(tickers)
            print(f"\n  Union size after {name}: {len(all_tickers)}")
        except Exception as e:
            print(f"\n  [ERROR] Source '{name}' failed: {e}")

    # Clean tickers
    cleaned = set()
    for t in all_tickers:
        t = t.strip().upper()
        # Remove suffixes
        if "." in t:
            t = t.split(".")[0]
        # Valid tickers: 1-5 alpha characters, optionally with . or - suffix
        if t and len(t) <= 6 and t.isascii() and t.replace("-", "").replace(".", "").isalpha():
            cleaned.add(t)

    # Remove common non-stock symbols
    exclude = {"", "A", "AA", "AAA", "AAAA", "AAAAA", "ETF", "ETFS", "FUND", "INC", "CORP",
               "LTD", "LLC", "LP", "GP", "TRUST", "UNIT", "NOTE", "DUE", "MAT", "SHS",
               "CL", "COM", "SER", "WTS", "WT", "RT", "WI", "PR", "CV"}
    cleaned -= exclude

    # Remove numbers
    cleaned = {t for t in cleaned if not t.isdigit()}

    return sorted(cleaned)


# ---------------------------------------------------------------------------
# 6. Download data with yfinance
# ---------------------------------------------------------------------------
def download_batch(tickers_batch, start, end, max_retries=MAX_RETRIES):
    """Download a batch of tickers with retries."""
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers_batch,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                actions=False,
                group_by="ticker",
                threads=True,
            )
            return data
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [RETRY] Batch failed (attempt {attempt+1}): {e}")
                time.sleep(2)
            else:
                print(f"  [FAIL] Batch failed after {max_retries} attempts: {e}")
                return None


def download_all(tickers, start, end, batch_size=BATCH_SIZE):
    """Download all tickers in batches, returning price and volume DataFrames."""
    total = len(tickers)
    batches = [tickers[i:i+batch_size] for i in range(0, total, batch_size)]
    print(f"\n{'='*70}")
    print(f"DOWNLOADING {total} TICKERS IN {len(batches)} BATCHES")
    print(f"{'='*70}")

    price_dfs = []
    volume_dfs = []
    successful = 0
    failed = 0

    for idx, batch in enumerate(batches):
        print(f"\r  Batch {idx+1}/{len(batches)} (tickers {idx*batch_size+1}-{min((idx+1)*batch_size, total)} of {total})...", end="")

        data = download_batch(list(batch), start, end)

        if data is None:
            failed += len(batch)
            time.sleep(SLEEP_SEC)
            continue

        # yfinance v0.2.0+ returns multi-level columns
        # For each ticker, extract Close and Volume
        for ticker in batch:
            ticker_str = str(ticker)
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    # MultiIndex columns: (Price, Ticker) or (Ticker, Price)
                    # Check orientation
                    if "Close" in data.columns.get_level_values(0):
                        # (Price, Ticker) orientation
                        if (ticker_str in data.columns.get_level_values(1)):
                            close_col = ("Close", ticker_str)
                            volume_col = ("Volume", ticker_str)
                            if close_col in data.columns and volume_col in data.columns:
                                p = data[close_col].dropna()
                                v = data[volume_col].dropna()
                                if len(p) > 0:
                                    p.name = ticker_str
                                    price_dfs.append(p)
                                if len(v) > 0:
                                    v.name = ticker_str
                                    volume_dfs.append(v)
                                successful += 1
                                continue
                    elif ticker_str in data.columns.get_level_values(0):
                        # (Ticker, Price) orientation
                        close_col = (ticker_str, "Close")
                        volume_col = (ticker_str, "Volume")
                        if close_col in data.columns and volume_col in data.columns:
                            p = data[close_col].dropna()
                            v = data[volume_col].dropna()
                            if len(p) > 0:
                                p.name = ticker_str
                                price_dfs.append(p)
                            if len(v) > 0:
                                v.name = ticker_str
                                volume_dfs.append(v)
                            successful += 1
                            continue
                else:
                    # Single ticker response (data is a single DataFrame)
                    if "Close" in data.columns and "Volume" in data.columns:
                        p = data["Close"].dropna()
                        v = data["Volume"].dropna()
                        if len(p) > 0:
                            p.name = ticker_str
                            price_dfs.append(p)
                        if len(v) > 0:
                            v.name = ticker_str
                            volume_dfs.append(v)
                        successful += 1
                        continue

                # If we get here, could not find data
                failed += 1

            except Exception:
                failed += 1

        time.sleep(SLEEP_SEC)

    print(f"\n  Done. Successful: {successful}, Failed: {failed}")

    # Merge into DataFrames
    print("\nMerging price data...")
    if price_dfs:
        price_df = pd.concat(price_dfs, axis=1, join="outer")
        price_df.index = pd.to_datetime(price_df.index)
        price_df = price_df.sort_index()
        price_df = price_df.loc["2000-01-01":"2024-12-31"]
    else:
        price_df = pd.DataFrame()

    print("Merging volume data...")
    if volume_dfs:
        volume_df = pd.concat(volume_dfs, axis=1, join="outer")
        volume_df.index = pd.to_datetime(volume_df.index)
        volume_df = volume_df.sort_index()
        volume_df = volume_df.loc["2000-01-01":"2024-12-31"]
    else:
        volume_df = pd.DataFrame()

    # Sort columns
    if not price_df.empty:
        price_df = price_df[sorted(price_df.columns)]
    if not volume_df.empty:
        volume_df = volume_df[sorted(volume_df.columns)]

    return price_df, volume_df, successful, failed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*70)
    print("COLLECTING TICKERS FROM ALL SOURCES")
    print("="*70)

    tickers = collect_all_tickers()

    print(f"\n{'='*70}")
    print(f"TOTAL UNIQUE TICKERS COLLECTED: {len(tickers)}")
    print(f"{'='*70}")

    if len(tickers) > 0:
        price_df, volume_df, successful, failed = download_all(tickers, START, END)

        print(f"\n{'='*70}")
        print("SAVING TO PARQUET")
        print(f"{'='*70}")

        print(f"\nPrice DataFrame shape: {price_df.shape}")
        print(f"Volume DataFrame shape: {volume_df.shape}")

        if not price_df.empty:
            price_df.to_parquet(PRICE_PARQUET, index=True)
            print(f"Saved price data: {PRICE_PARQUET}")
        else:
            print("No price data to save")

        if not volume_df.empty:
            volume_df.to_parquet(VOLUME_PARQUET, index=True)
            print(f"Saved volume data: {VOLUME_PARQUET}")
        else:
            print("No volume data to save")

        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"  Tickers attempted: {len(tickers)}")
        print(f"  Successfully downloaded: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Date range: {START} to {END}")

        # File sizes
        for path in [PRICE_PARQUET, VOLUME_PARQUET]:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"  {os.path.basename(path)}: {size_mb:.2f} MB")
            else:
                print(f"  {path}: NOT FOUND")

        print(f"\n{'='*70}")
        print("DONE")
        print(f"{'='*70}")
    else:
        print("\nERROR: No tickers collected. Cannot proceed.")
        sys.exit(1)
