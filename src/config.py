"""Project configuration."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

DATA_FILE = DATA_DIR / "marketing_campaign.xlsx"
TARGET = "response"
RANDOM_STATE = 42
TEST_SIZE = 0.20

DROP_COLUMNS = ["id", "z_costcontact", "z_cost_contact", "z_revenue"]
NUMERIC_OUTLIER_COLUMNS = [
    "age", "income", "totalkids", "recency", "mntwines", "mntfruits",
    "mntmeatproducts", "mntfishproducts", "mntsweetproducts", "mntgoldprods",
    "numdealspurchases", "numwebpurchases", "numcatalogpurchases",
    "numstorepurchases", "numwebvisitsmonth", "years_enrolled"
]
