# config.py
import logging
from dotenv import load_dotenv

# Load environment variables (from .env file if you use one)
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# -------------------------------------------------------------------------------------
# Define the TEAMS Dictionary
# -------------------------------------------------------------------------------------
TEAMS = {
    'hawks': {'full_name': 'Atlanta Hawks', 'logo': 'hawks_logo.png'},
    'nets': {'full_name': 'Brooklyn Nets', 'logo': 'nets_logo.png'},
    'celtics': {'full_name': 'Boston Celtics', 'logo': 'celtics_logo.png'},
    'hornets': {'full_name': 'Charlotte Hornets', 'logo': 'hornets_logo.png'},
    'bulls': {'full_name': 'Chicago Bulls', 'logo': 'bulls_logo.png'},
    'cavaliers': {'full_name': 'Cleveland Cavaliers', 'logo': 'cavaliers_logo.png'},
    'mavericks': {'full_name': 'Dallas Mavericks', 'logo': 'mavericks_logo.png'},
    'nuggets': {'full_name': 'Denver Nuggets', 'logo': 'nuggets_logo.png'},
    'pistons': {'full_name': 'Detroit Pistons', 'logo': 'pistons_logo.png'},
    'warriors': {'full_name': 'Golden State Warriors', 'logo': 'warriors_logo.png'},
    'rockets': {'full_name': 'Houston Rockets', 'logo': 'rockets_logo.png'},
    'pacers': {'full_name': 'Indiana Pacers', 'logo': 'pacers_logo.png'},
    'clippers': {'full_name': 'Los Angeles Clippers', 'logo': 'clippers_logo.png'},
    'lakers': {'full_name': 'Los Angeles Lakers', 'logo': 'lakers_logo.png'},
    'grizzlies': {'full_name': 'Memphis Grizzlies', 'logo': 'grizzlies_logo.png'},
    'heat': {'full_name': 'Miami Heat', 'logo': 'heat_logo.png'},
    'bucks': {'full_name': 'Milwaukee Bucks', 'logo': 'bucks_logo.png'},
    'timberwolves': {'full_name': 'Minnesota Timberwolves', 'logo': 'timberwolves_logo.png'},
    'pelicans': {'full_name': 'New Orleans Pelicans', 'logo': 'pelicans_logo.png'},
    'knicks': {'full_name': 'New York Knicks', 'logo': 'knicks_logo.png'},
    'thunder': {'full_name': 'Oklahoma City Thunder', 'logo': 'thunder_logo.png'},
    'magic': {'full_name': 'Orlando Magic', 'logo': 'magic_logo.png'},
    '76ers': {'full_name': 'Philadelphia 76ers', 'logo': '76ers_logo.png'},
    'suns': {'full_name': 'Phoenix Suns', 'logo': 'suns_logo.png'},
    'blazers': {'full_name': 'Portland Trail Blazers', 'logo': 'blazers_logo.png'},
    'kings': {'full_name': 'Sacramento Kings', 'logo': 'kings_logo.png'},
    'spurs': {'full_name': 'San Antonio Spurs', 'logo': 'spurs_logo.png'},
    'raptors': {'full_name': 'Toronto Raptors', 'logo': 'raptors_logo.png'},
    'jazz': {'full_name': 'Utah Jazz', 'logo': 'jazz_logo.png'},
    'wizards': {'full_name': 'Washington Wizards', 'logo': 'wizards_logo.png'}
}


DYNAMODB_TABLE_USERS = 'StreamlitUsers'      # DynamoDB table for user data
DYNAMODB_TABLE_DATA = 'ProcessingQueue'      # DynamoDB table for job data


# -------------------------------------------------------------------------------------
# Define DynamoDB Table Names as Constants
# -------------------------------------------------------------------------------------
BUCKET_NAME = "pivotalbucket"

# config.py
FPS = 60  # Update to match your data's frame rate
BASKET_HEIGHT_IN = 120
RIM_POSITION = {'x': 0, 'y': BASKET_HEIGHT_IN}

# For spin KPIs (example ranges)
SPIN_KPI_RANGES = {
    'spin_magnitude': (0, 500),
    'spin_consistency': (0, 100),
    'spin_duration': (0, 10)
}